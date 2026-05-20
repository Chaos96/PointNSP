"""Multi-Scale VQVAE for 3D point cloud generation (Algorithms 1 & 2).

Implements the two-stage framework encoder/decoder with:
- PVCNN encoder for per-point features
- FPS-based LoD hierarchy
- Shared codebook with straight-through VQ
- Per-scale Phi refinement networks
- PU-Net upsampling
- 6-layer MLP decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvcnn import PVCNN
from .fps import farthest_point_sampling, build_lod_sequence
from .upsampling import punet_upsample


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer Distance between two point clouds.

    Uses torch.cdist for efficient batched computation.

    Args:
        x: (B, N, 3)
        y: (B, M, 3)
    Returns:
        Scalar mean CD loss.
    """
    dist = torch.cdist(x, y)  # (B, N, M)
    min_x_to_y = dist.min(dim=2)[0]  # (B, N)
    min_y_to_x = dist.min(dim=1)[0]  # (B, M)
    return min_x_to_y.mean() + min_y_to_x.mean()


def earth_movers_distance_approx(
    x: torch.Tensor, y: torch.Tensor, n_iters: int = 50, reg: float = 0.05
) -> torch.Tensor:
    """Approximate Earth Mover's Distance via Sinkhorn algorithm.

    Memory-efficient: processes per-sample.

    Args:
        x: (B, N, 3)
        y: (B, M, 3)  (typically M == N)
        n_iters: number of Sinkhorn iterations
        reg: entropic regularization
    Returns:
        Scalar EMD approximation.
    """
    B, N, _ = x.shape
    M = y.shape[1]
    emd = 0.0

    for i in range(B):
        cost = torch.cdist(x[i:i+1], y[i:i+1]).squeeze(0)  # (N, M)
        cost = cost.clamp(min=0.0)

        K = torch.exp(-cost / reg)
        u = torch.ones(N, 1, device=x.device) / N
        for _ in range(n_iters):
            v = 1.0 / (M * (K.t() @ u + 1e-8))
            u = 1.0 / (N * (K @ v + 1e-8))
        T = u * K * v.t()  # (N, M)
        emd += (T * cost).sum()

    return emd / B


# ---------------------------------------------------------------------------
# Vector Quantizer with straight-through and EMA tracking
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Shared codebook with straight-through estimator.

    Follows SAR3D patterns:
    - Commitment + codebook loss (Eq. 7)
    - Straight-through gradient: f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
    - EMA hit-count tracking per scale for utilization monitoring
    """

    def __init__(self, codebook_size: int, embedding_dim: int, beta: float = 0.25,
                 num_scales: int = 1):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA hit tracking per scale
        self.register_buffer(
            'ema_vocab_hit_SV',
            torch.zeros(num_scales, codebook_size),
        )
        self.record_hit: int = 0

    def forward(self, z: torch.Tensor, scale_idx: int = 0):
        """Quantize input features.

        Args:
            z: (B, s_k, D) features to quantize.
            scale_idx: current scale index for EMA tracking.
        Returns:
            quantized: (B, s_k, D) quantized features (straight-through).
            vq_loss: scalar VQ loss.
            indices: (B, s_k) codebook indices.
        """
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*N, D)
        z_no_grad = z_flat.detach()

        # L2 distance to codebook
        d = (
            z_no_grad.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1, keepdim=False)
            - 2.0 * z_no_grad @ self.embedding.weight.t()
        )
        indices = d.argmin(dim=1)  # (B*N,)

        # EMA tracking
        if self.training:
            hit_V = indices.bincount(minlength=self.codebook_size).float()
            if self.record_hit == 0:
                self.ema_vocab_hit_SV[scale_idx].copy_(hit_V)
            elif self.record_hit < 100:
                self.ema_vocab_hit_SV[scale_idx].mul_(0.9).add_(hit_V, alpha=0.1)
            else:
                self.ema_vocab_hit_SV[scale_idx].mul_(0.99).add_(hit_V, alpha=0.01)
            self.record_hit += 1

        # Lookup
        quantized_flat = self.embedding(indices)  # (B*N, D)
        quantized = quantized_flat.reshape(B, N, D)

        # VQ loss (SAR3D style): commitment + codebook
        f_hat_data = quantized.detach()
        f_no_grad_full = z.detach()
        vq_loss = (
            F.mse_loss(f_hat_data, z) * self.beta
            + F.mse_loss(quantized, f_no_grad_full)
        )

        # Straight-through estimator (SAR3D style)
        quantized = (quantized.data - z.detach()).add_(z)

        indices = indices.reshape(B, N)
        return quantized, vq_loss, indices


# ---------------------------------------------------------------------------
# Phi Network — per-point MLP refinement (permutation equivariant)
# ---------------------------------------------------------------------------

class PhiNetwork(nn.Module):
    """Per-point MLP refinement network (permutation equivariant).

    Applies a small residual MLP to each point feature independently,
    following the Phi refinement pattern from SAR3D: h * (1 - ratio) + mlp(h) * ratio.
    """

    def __init__(self, dim: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        return x * (1 - self.ratio) + self.net(x) * self.ratio


# ---------------------------------------------------------------------------
# Multi-Scale VQVAE (Algorithms 1 & 2)
# ---------------------------------------------------------------------------

class MultiScaleVQVAE(nn.Module):
    """Multi-Scale VQVAE implementing Algorithm 1 (encode) & Algorithm 2 (decode).

    Args:
        hidden_dim: Feature dimension (PVCNN output and codebook embedding dim).
        codebook_size: Number of entries in the shared codebook.
        num_points: Total number of points in the input point cloud (s_K).
        scale_points: Ascending list of LoD sizes, e.g. [4, 16, 64, 256, 2048].
                      Last element must equal num_points.
        pvcnn_layers: Number of PVConv blocks in PVCNN encoder.
        voxel_resolution: Voxel grid resolution for PVCNN.
        beta: Commitment loss coefficient for VQ.
        phi_ratio: Residual ratio for PhiNetwork.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        codebook_size: int = 8192,
        num_points: int = 2048,
        scale_points: list = None,
        pvcnn_layers: int = 4,
        voxel_resolution: int = 32,
        beta: float = 0.25,
        phi_ratio: float = 0.5,
    ):
        super().__init__()
        if scale_points is None:
            scale_points = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        assert scale_points[-1] == num_points, (
            f"Last scale ({scale_points[-1]}) must equal num_points ({num_points})"
        )
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_points = num_points
        self.scale_points = scale_points
        self.num_scales = len(scale_points)

        # Encoder
        self.encoder = PVCNN(
            hidden_dim=hidden_dim,
            num_layers=pvcnn_layers,
            voxel_resolution=voxel_resolution,
        )

        # Shared VQ layer
        self.vq_layer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=hidden_dim,
            beta=beta,
            num_scales=self.num_scales,
        )

        # Per-scale Phi refinement networks
        self.phi = nn.ModuleList([
            PhiNetwork(hidden_dim, ratio=phi_ratio) for _ in range(self.num_scales)
        ])

        # Decoder: 6-layer MLP (5 hidden Linear+ReLU + 1 output Linear -> 3)
        decoder_layers = []
        for _ in range(5):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dim, 3))
        self.decoder = nn.Sequential(*decoder_layers)

    # ----- helpers -----

    def query(self, features: torch.Tensor, lod_indices_k: torch.Tensor) -> torch.Tensor:
        """Gather features at scale k's point positions.

        Args:
            features: (B, N, D) full-resolution features.
            lod_indices_k: (B, s_k) indices into the N dimension.
        Returns:
            (B, s_k, D) gathered features.
        """
        idx = lod_indices_k.unsqueeze(-1).expand(-1, -1, features.shape[-1])  # (B, s_k, D)
        return torch.gather(features, 1, idx)

    # ----- Algorithm 1: Encoder -----

    def encode(self, x: torch.Tensor):
        """Algorithm 1 — encode point cloud to multi-scale token sequences.

        Args:
            x: (B, N, 3) input point cloud.
        Returns:
            token_lists: list of K tensors, each (B, s_k) of long indices.
            vq_losses: list of K scalar VQ losses.
            f_residual: (B, N, D) final residual (for debugging).
        """
        B, N, _ = x.shape
        s_K = self.scale_points[-1]

        # Step 1: PVCNN encoder
        f = self.encoder(x)  # (B, N, hidden_dim)

        # Build LoD hierarchy
        _, lod_indices = build_lod_sequence(x, self.scale_points)

        # Running residual
        f_residual = f  # (B, N, D)

        token_lists = []
        vq_losses = []

        for k in range(self.num_scales):
            # Query features at scale k's positions
            f_k = self.query(f_residual, lod_indices[k])  # (B, s_k, D)

            # Quantize
            z_k, vq_loss, q_k = self.vq_layer(f_k, scale_idx=k)
            token_lists.append(q_k)
            vq_losses.append(vq_loss)

            # Upsample to full resolution
            z_k_up = punet_upsample(z_k, s_K)  # (B, s_K, D)

            # Phi refinement
            f_tilde_k = self.phi[k](z_k_up)  # (B, s_K, D)

            # Update running residual
            f_residual = f_residual - f_tilde_k

        return token_lists, vq_losses, f_residual

    # ----- Algorithm 2: Decoder -----

    def decode(self, token_lists: list) -> torch.Tensor:
        """Algorithm 2 — decode token sequences to point cloud.

        Args:
            token_lists: list of K tensors, each (B, s_k) of long indices.
        Returns:
            x_hat: (B, N, 3) reconstructed point cloud.
        """
        s_K = self.scale_points[-1]
        B = token_lists[0].shape[0]
        D = self.hidden_dim
        device = token_lists[0].device

        f_hat = torch.zeros(B, s_K, D, device=device)

        for k in range(self.num_scales):
            # Codebook lookup
            z_k = self.vq_layer.embedding(token_lists[k])  # (B, s_k, D)

            # Upsample to full resolution
            z_k_up = punet_upsample(z_k, s_K)  # (B, s_K, D)

            # Phi refinement and accumulate
            f_hat = f_hat + self.phi[k](z_k_up)

        # MLP decoder
        x_hat = self.decoder(f_hat)  # (B, s_K, 3)
        return x_hat

    # ----- Forward -----

    def forward(self, x: torch.Tensor):
        """Full forward: encode + decode + compute losses.

        Args:
            x: (B, N, 3) input point cloud.
        Returns:
            x_hat: (B, N, 3) reconstructed point cloud.
            total_loss: scalar total loss.
            recon_loss: scalar reconstruction loss (CD + EMD).
            vq_loss: scalar VQ loss (sum over scales).
        """
        token_lists, vq_losses, _ = self.encode(x)
        x_hat = self.decode(token_lists)

        # Reconstruction loss (Eq. 7): L_CD + L_EMD
        cd_loss = chamfer_distance(x, x_hat)
        emd_loss = earth_movers_distance_approx(x, x_hat)
        recon_loss = cd_loss + emd_loss

        # VQ loss (sum over scales)
        vq_loss = sum(vq_losses)

        total_loss = recon_loss + vq_loss

        return x_hat, total_loss, recon_loss, vq_loss

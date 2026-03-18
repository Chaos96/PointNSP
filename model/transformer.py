"""Autoregressive Transformer for Next-Scale LoD Prediction (Stage 2).

Implements the PointNSP transformer (Sec 3.3) that autoregressively predicts
codebook tokens scale-by-scale. Within each scale, tokens attend to each other
bidirectionally (VAR-style). Across scales, a block-wise causal mask ensures
scale k can only attend to scales 1..k.

Key equations:
- Eq. 23: u_k^i = W_U z_k^i + p_k^i + s_k^i  (query)
          v_k^i = W_V z_k^i + p_k^i + s_k^i  (key)
          value = W_val z_k^i  (no pos/scale)
- Eq. 8:  Position-aware soft mask (intra-scale only)
- Eq. 9:  Intermediate structure decoding (inference only)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import BAPEPositionalEncoding, ScaleEmbedding
from .masking import build_blockwise_causal_mask, build_position_aware_soft_mask


class PointNSPTransformerBlock(nn.Module):
    """Single transformer block with custom attention following Eq. 23.

    - Q/K receive positional + scale embeddings BEFORE projection.
    - V does NOT get positional or scale embeddings.
    - Block-wise causal mask applied to attention scores.
    - Position-aware soft mask applied to intra-scale attention.
    - Pre-norm architecture with LayerNorm and GELU FFN.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Q/K/V projections (Eq. 23)
        self.W_Q = nn.Linear(d_model, d_model)  # projects (token_emb + pos + scale)
        self.W_K = nn.Linear(d_model, d_model)  # projects (token_emb + pos + scale)
        self.W_V = nn.Linear(d_model, d_model)  # projects token_emb only

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Position-aware soft mask projection (Eq. 8)
        self.W_p = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        scale_emb: torch.Tensor,
        causal_mask: torch.Tensor,
        scale_points: list,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) token embeddings.
            pos_emb: (B, T, d_model) BAPE positional encodings.
            scale_emb: (B, T, d_model) scale embeddings (broadcast per scale).
            causal_mask: (T, T) block-wise causal mask (1=attend, 0=mask).
            scale_points: list of ints for building intra-scale soft masks.
        Returns:
            (B, T, d_model) output after attention + FFN.
        """
        B, T, D = x.shape

        # ---------- Self-Attention with Pre-Norm ----------
        residual = x
        x_norm = self.norm1(x)

        # Eq. 23: Q/K get pos + scale; V does not
        q_input = x_norm + pos_emb + scale_emb  # (B, T, D)
        k_input = x_norm + pos_emb + scale_emb  # (B, T, D)
        v_input = x_norm                          # (B, T, D)

        Q = self.W_Q(q_input)  # (B, T, D)
        K = self.W_K(k_input)  # (B, T, D)
        V = self.W_V(v_input)  # (B, T, D)

        # Reshape to multi-head: (B, nhead, T, head_dim)
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, nhead, T, T)

        # Apply block-wise causal mask (0 -> -inf)
        causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn_scores = attn_scores.masked_fill(causal_mask_expanded == 0, float('-inf'))

        # Apply position-aware soft mask (Eq. 8) to intra-scale blocks
        soft_mask_log = self._build_soft_mask_matrix(pos_emb, scale_points)  # (B, T, T)
        attn_scores = attn_scores + soft_mask_log.unsqueeze(1)  # broadcast over heads

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)  # (B, nhead, T, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out

        # ---------- FFN with Pre-Norm ----------
        residual = x
        x = residual + self.ffn_dropout(self.ffn(self.norm2(x)))

        return x

    def _build_soft_mask_matrix(
        self, pos_emb: torch.Tensor, scale_points: list
    ) -> torch.Tensor:
        """Build the full T x T soft mask matrix from intra-scale soft masks.

        For intra-scale blocks: add log(M_k^p) to attention scores.
        For inter-scale blocks: add 0 (no modification).
        """
        B, T, D = pos_emb.shape
        soft_mask = torch.zeros(B, T, T, device=pos_emb.device)

        offset = 0
        for s_k in scale_points:
            P_k = pos_emb[:, offset:offset + s_k, :]  # (B, s_k, D)
            M_k = build_position_aware_soft_mask(P_k, self.W_p)  # (B, s_k, s_k)
            # log(softmax) values; clamp to avoid log(0)
            log_M_k = torch.log(M_k.clamp(min=1e-8))
            soft_mask[:, offset:offset + s_k, offset:offset + s_k] = log_M_k
            offset += s_k

        return soft_mask


class PointNSPTransformer(nn.Module):
    """Autoregressive Transformer for next-scale LoD prediction.

    Uses teacher forcing during training: all ground-truth token embeddings
    are fed as input, and block-wise causal masking ensures scale k can only
    attend to scales 1..k (autoregressive across scales, bidirectional within).

    Args:
        codebook_size: Number of codebook entries (vocabulary size).
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer blocks.
        scale_points: List of ints [s_1, ..., s_K].
        num_scales: Number of scales K.
        lam: Lambda for BAPE positional encoding.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        codebook_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        scale_points: list,
        num_scales: int,
        lam: float = 1000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.scale_points = scale_points
        self.num_scales = num_scales
        self.total_seq_len = sum(scale_points)

        # Token embedding: codebook index -> d_model
        self.token_embedding = nn.Embedding(codebook_size, d_model)

        # Learnable [start] token embedding
        self.start_token = nn.Parameter(torch.randn(d_model) * 0.02)

        # Positional encoding (BAPE from 3D coordinates)
        self.bape = BAPEPositionalEncoding(d_model, lam=lam)

        # Scale embedding
        self.scale_embedding = ScaleEmbedding(num_scales, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PointNSPTransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection to logits
        self.output_proj = nn.Linear(d_model, codebook_size)

        # Intermediate structure decoder (Eq. 9): for generating 3D coords at inference
        self.intermediate_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def _build_input_embeddings(
        self, tokens: List[torch.Tensor]
    ) -> torch.Tensor:
        """Build shifted input embeddings for teacher forcing.

        Scale 1 positions: [start] token expanded to s_1 positions.
        Scale k>1 positions: embed(q_{k-1}) adapted to s_k positions via repeat.
        Actually, for simplicity with block-wise causal mask, we embed ALL GT
        tokens and rely on the causal mask. But scale 1 input must be [start].

        Args:
            tokens: list of K tensors, each (B, s_k) long indices.
        Returns:
            (B, T, d_model) input embeddings.
        """
        B = tokens[0].shape[0]
        device = tokens[0].device
        embeddings = []

        for k in range(self.num_scales):
            s_k = self.scale_points[k]
            if k == 0:
                # Scale 1: use [start] token expanded to s_1 positions
                start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, s_k, -1)
                embeddings.append(start)
            else:
                # Scale k>1: embed tokens from scale k-1, adapt to s_k positions
                prev_tokens = tokens[k - 1]  # (B, s_{k-1})
                prev_emb = self.token_embedding(prev_tokens)  # (B, s_{k-1}, d_model)
                s_prev = self.scale_points[k - 1]
                # Repeat/interleave to fill s_k positions
                if s_k >= s_prev and s_k % s_prev == 0:
                    r = s_k // s_prev
                    adapted = prev_emb.unsqueeze(2).expand(B, s_prev, r, self.d_model)
                    adapted = adapted.reshape(B, s_k, self.d_model)
                else:
                    # Fallback: interpolate
                    # (B, s_prev, D) -> (B, D, s_prev) for F.interpolate
                    prev_t = prev_emb.transpose(1, 2)
                    adapted = F.interpolate(prev_t, size=s_k, mode='linear', align_corners=False)
                    adapted = adapted.transpose(1, 2)
                embeddings.append(adapted)

        return torch.cat(embeddings, dim=1)  # (B, T, d_model)

    def _build_pos_and_scale_embeddings(
        self, coords: List[torch.Tensor]
    ) -> tuple:
        """Build BAPE positional encodings and scale embeddings for all positions.

        Args:
            coords: list of K tensors, each (B, s_k, 3) coordinates.
        Returns:
            pos_emb: (B, T, d_model) BAPE positional encodings.
            scale_emb: (B, T, d_model) scale embeddings.
        """
        B = coords[0].shape[0]
        pos_parts = []
        scale_parts = []

        for k in range(self.num_scales):
            s_k = self.scale_points[k]
            # BAPE from 3D coords
            p_k = self.bape(coords[k])  # (B, s_k, d_model)
            pos_parts.append(p_k)

            # Scale embedding: same for all tokens in scale k
            s_emb = self.scale_embedding(k)  # (d_model,)
            s_emb_expanded = s_emb.unsqueeze(0).unsqueeze(0).expand(B, s_k, -1)
            scale_parts.append(s_emb_expanded)

        pos_emb = torch.cat(pos_parts, dim=1)    # (B, T, d_model)
        scale_emb = torch.cat(scale_parts, dim=1)  # (B, T, d_model)
        return pos_emb, scale_emb

    def forward(
        self,
        tokens: List[torch.Tensor],
        coords: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Forward pass with teacher forcing.

        The input sequence uses shifted tokens (scale k receives tokens from
        scale k-1, scale 1 receives [start]). Block-wise causal mask ensures
        autoregression across scales.

        Args:
            tokens: list of K tensors, each (B, s_k) of long codebook indices.
            coords: list of K tensors, each (B, s_k, 3) point coordinates.
        Returns:
            logits_per_scale: list of K tensors, each (B, s_k, codebook_size).
        """
        device = tokens[0].device

        # Build input token embeddings (shifted for teacher forcing)
        x = self._build_input_embeddings(tokens)  # (B, T, d_model)

        # Build positional and scale embeddings
        pos_emb, scale_emb = self._build_pos_and_scale_embeddings(coords)

        # Block-wise causal mask
        causal_mask = build_blockwise_causal_mask(self.scale_points, device=device)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, pos_emb, scale_emb, causal_mask, self.scale_points)

        # Final norm + output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)  # (B, T, codebook_size)

        # Split logits per scale
        logits_per_scale = []
        offset = 0
        for k in range(self.num_scales):
            s_k = self.scale_points[k]
            logits_per_scale.append(logits[:, offset:offset + s_k, :])
            offset += s_k

        return logits_per_scale

    def compute_loss(
        self,
        tokens: List[torch.Tensor],
        coords: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-entropy loss (Eq. in Sec 3.3).

        L_k = (1/s_k) * sum_i CE(logits_k^i, q_k^i)
        L_total = (1/K) * sum_k L_k

        Args:
            tokens: list of K tensors, each (B, s_k) long indices.
            coords: list of K tensors, each (B, s_k, 3) coordinates.
        Returns:
            Scalar CE loss averaged over scales.
        """
        logits_per_scale = self.forward(tokens, coords)

        total_loss = 0.0
        for k in range(self.num_scales):
            logits_k = logits_per_scale[k]  # (B, s_k, codebook_size)
            targets_k = tokens[k]           # (B, s_k)
            B, s_k, C = logits_k.shape
            # Reshape for cross_entropy: (B*s_k, C) vs (B*s_k,)
            loss_k = F.cross_entropy(
                logits_k.reshape(-1, C), targets_k.reshape(-1), reduction='mean'
            )
            total_loss += loss_k

        total_loss = total_loss / self.num_scales
        return total_loss

    @torch.no_grad()
    def generate(
        self,
        vqvae,
        num_samples: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation: predict tokens scale-by-scale.

        1. Scale 1: use [start] as input, predict s_1 tokens.
        2. Scale k>1: use generated tokens from scales 1..k-1 as context,
           predict s_k tokens simultaneously (bidirectional intra-scale).
        3. Decode intermediate coords via Eq. 9 for BAPE at next scale.

        Args:
            vqvae: Pretrained MultiScaleVQVAE (for codebook and Phi networks).
            num_samples: Batch size (number of samples to generate).
            device: Torch device.
            temperature: Sampling temperature.
        Returns:
            point_cloud: (num_samples, s_K, 3) generated point clouds.
        """
        self.eval()
        B = num_samples
        T = self.total_seq_len

        generated_tokens = []  # list of K tensors, each (B, s_k)
        generated_coords = []  # list of K tensors, each (B, s_k, 3)

        for k in range(self.num_scales):
            s_k = self.scale_points[k]

            # Build the full input sequence up to scale k
            # Scales 1..k-1: use generated token embeddings
            # Scale k: use placeholder (zeros or [start] for k=0)
            input_parts = []
            pos_parts = []
            scale_parts = []

            for j in range(k):
                s_j = self.scale_points[j]
                if j == 0:
                    # Scale 1 input is always [start]
                    start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, s_j, -1)
                    input_parts.append(start)
                else:
                    # Use generated tokens from scale j-1 as input for scale j
                    prev_emb = self.token_embedding(generated_tokens[j - 1])
                    s_prev = self.scale_points[j - 1]
                    if s_j >= s_prev and s_j % s_prev == 0:
                        r = s_j // s_prev
                        adapted = prev_emb.unsqueeze(2).expand(B, s_prev, r, self.d_model)
                        adapted = adapted.reshape(B, s_j, self.d_model)
                    else:
                        prev_t = prev_emb.transpose(1, 2)
                        adapted = F.interpolate(prev_t, size=s_j, mode='linear', align_corners=False)
                        adapted = adapted.transpose(1, 2)
                    input_parts.append(adapted)

                # Positional encoding from generated coords
                p_j = self.bape(generated_coords[j])
                pos_parts.append(p_j)
                s_emb_j = self.scale_embedding(j).unsqueeze(0).unsqueeze(0).expand(B, s_j, -1)
                scale_parts.append(s_emb_j)

            # Current scale k input
            if k == 0:
                start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, s_k, -1)
                input_parts.append(start)
            else:
                # Use generated tokens from scale k-1 as shifted input
                prev_emb = self.token_embedding(generated_tokens[k - 1])
                s_prev = self.scale_points[k - 1]
                if s_k >= s_prev and s_k % s_prev == 0:
                    r = s_k // s_prev
                    adapted = prev_emb.unsqueeze(2).expand(B, s_prev, r, self.d_model)
                    adapted = adapted.reshape(B, s_k, self.d_model)
                else:
                    prev_t = prev_emb.transpose(1, 2)
                    adapted = F.interpolate(prev_t, size=s_k, mode='linear', align_corners=False)
                    adapted = adapted.transpose(1, 2)
                input_parts.append(adapted)

            # For scale k's positions: we need coords for BAPE.
            # Use intermediate structure decoding (Eq. 9) from accumulated tokens.
            if k == 0:
                # No prior tokens; use zero coords for scale 1
                coords_k = torch.zeros(B, s_k, 3, device=device)
            else:
                # Eq. 9: decode intermediate structure from generated tokens so far
                coords_k = self._decode_intermediate_coords(
                    vqvae, generated_tokens, k, device
                )

            p_k = self.bape(coords_k)
            pos_parts.append(p_k)
            s_emb_k = self.scale_embedding(k).unsqueeze(0).unsqueeze(0).expand(B, s_k, -1)
            scale_parts.append(s_emb_k)

            # Concatenate all parts
            x = torch.cat(input_parts, dim=1)       # (B, T_k, d_model)
            pos_emb = torch.cat(pos_parts, dim=1)   # (B, T_k, d_model)
            scale_emb = torch.cat(scale_parts, dim=1)

            # Build causal mask for scales 1..k
            current_scale_points = self.scale_points[:k + 1]
            causal_mask = build_blockwise_causal_mask(current_scale_points, device=device)

            # Pass through transformer
            for block in self.blocks:
                x = block(x, pos_emb, scale_emb, causal_mask, current_scale_points)

            x = self.final_norm(x)
            logits = self.output_proj(x)  # (B, T_k, codebook_size)

            # Extract logits for scale k (last s_k positions)
            logits_k = logits[:, -s_k:, :]  # (B, s_k, codebook_size)

            # Sample tokens
            if temperature > 0:
                probs = F.softmax(logits_k / temperature, dim=-1)
                sampled = torch.multinomial(
                    probs.reshape(-1, self.codebook_size), 1
                ).reshape(B, s_k)
            else:
                sampled = logits_k.argmax(dim=-1)

            generated_tokens.append(sampled)
            generated_coords.append(coords_k)

        # Final decode: use VQVAE decoder to get point cloud from all tokens
        point_cloud = vqvae.decode(generated_tokens)  # (B, s_K, 3)
        return point_cloud

    def _decode_intermediate_coords(
        self,
        vqvae,
        generated_tokens: list,
        current_scale: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Eq. 9: Intermediate structure decoding for BAPE coordinates.

        X_k = D(sum_{m=1}^{k} phi_m(upsample(z_m, s_m)))

        Uses the VQVAE's codebook, phi networks, and the intermediate decoder.

        Args:
            vqvae: Pretrained VQVAE model.
            generated_tokens: list of tensors with tokens generated so far.
            current_scale: current scale index k (0-based).
            device: torch device.
        Returns:
            coords: (B, s_k, 3) predicted coordinates for current scale.
        """
        from .upsampling import punet_upsample

        B = generated_tokens[0].shape[0]
        s_k = self.scale_points[current_scale]

        # Accumulate features
        f_acc = torch.zeros(B, s_k, self.d_model, device=device)

        for m in range(current_scale):
            s_m = self.scale_points[m]
            # Codebook lookup
            z_m = vqvae.vq_layer.embedding(generated_tokens[m])  # (B, s_m, D_vqvae)

            # Upsample to s_k positions
            z_m_up = punet_upsample(z_m, s_k)  # (B, s_k, D_vqvae)

            # Phi refinement
            z_m_refined = vqvae.phi[m](z_m_up)  # (B, s_k, D_vqvae)

            # If VQVAE hidden dim differs from transformer d_model, we need
            # to handle this. For now, assume they match or truncate/pad.
            d_vqvae = z_m_refined.shape[-1]
            if d_vqvae == self.d_model:
                f_acc = f_acc + z_m_refined
            elif d_vqvae < self.d_model:
                f_acc[:, :, :d_vqvae] = f_acc[:, :, :d_vqvae] + z_m_refined
            else:
                f_acc = f_acc + z_m_refined[:, :, :self.d_model]

        # Decode to 3D coordinates using intermediate decoder
        coords = self.intermediate_decoder(f_acc)  # (B, s_k, 3)
        return coords

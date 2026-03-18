import torch
import torch.nn as nn
import math


class BAPEPositionalEncoding(nn.Module):
    """Base-lambda Position Encoding (BAPE) from Eq. 20-22.

    Converts 3D coordinates to positional encodings via:
    p = lambda^2 * z + lambda * y + x
    PE(p, 2i) = sin(p / 10000^(2i/d))
    PE(p, 2i+1) = cos(p / 10000^(2i/d))  # SAME divisor as sin
    """

    def __init__(self, d_model: int, lam: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        self.lam = lam
        # Precompute: 1 / 10000^(2i/d) for i = 0, 1, ..., d/2-1
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)  # (d_model/2,)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) point coordinates (x, y, z)
        Returns:
            pe: (B, N, d_model) positional encoding
        """
        x = coords[..., 0]  # (B, N)
        y = coords[..., 1]
        z = coords[..., 2]
        p = self.lam ** 2 * z + self.lam * y + x  # (B, N)
        p = p.unsqueeze(-1)  # (B, N, 1)

        pe = torch.zeros(*coords.shape[:2], self.d_model, device=coords.device)
        # Both sin and cos use the SAME div_term (critical per Eq. 21-22)
        pe[..., 0::2] = torch.sin(p * self.div_term)  # div_term = 1/10000^(2i/d)
        pe[..., 1::2] = torch.cos(p * self.div_term)
        return pe


class ScaleEmbedding(nn.Module):
    """Learnable scale embedding s_k (Supplementary Sec 7).

    One-hot-like embedding over K scales. All tokens within the same
    scale share the same embedding vector.
    """

    def __init__(self, num_scales: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_scales, d_model)

    def forward(self, scale_idx: int) -> torch.Tensor:
        """Returns (d_model,) embedding for scale index."""
        idx = torch.tensor(scale_idx, device=self.embedding.weight.device)
        return self.embedding(idx)

    def get_all(self) -> torch.Tensor:
        """Returns (num_scales, d_model) all scale embeddings."""
        return self.embedding.weight

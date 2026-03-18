import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    """Per-point MLP: Linear + BatchNorm + ReLU.

    Operates independently on each point, preserving permutation equivariance.
    Input shape: (B, N, C_in) -> Output shape: (B, N, C_out).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C_in)
        Returns:
            (B, N, C_out)
        """
        # Linear operates on last dim, so (B, N, C_in) -> (B, N, C_out)
        out = self.linear(x)
        # BatchNorm1d expects (B, C, N), so transpose, apply, transpose back
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        return out

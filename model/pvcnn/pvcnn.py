import torch
import torch.nn as nn

from .shared_mlp import SharedMLP
from .voxelization import Voxelization


class PVConvBlock(nn.Module):
    """One Point-Voxel Convolution block.

    Point branch (SharedMLP) and voxel branch (Voxelization) run in
    parallel on the same input features; their outputs are concatenated
    and fused back to the original hidden dimension via a linear layer.
    """

    def __init__(self, channels: int, voxel_resolution: int = 32):
        super().__init__()
        self.point_branch = SharedMLP(channels, channels)
        self.voxel_branch = Voxelization(channels, channels, resolution=voxel_resolution)
        # After concatenation the feature dim is 2*channels; fuse back.
        self.fusion = nn.Linear(channels * 2, channels)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, C)
            coords:   (B, N, 3)
        Returns:
            (B, N, C)
        """
        point_out = self.point_branch(features)        # (B, N, C)
        voxel_out = self.voxel_branch(features, coords)  # (B, N, C)
        fused = torch.cat([point_out, voxel_out], dim=-1)  # (B, N, 2C)
        return self.fusion(fused)  # (B, N, C)


class PVCNN(nn.Module):
    """Point-Voxel CNN encoder.

    Processes a raw point cloud (B, N, 3) and returns per-point features
    (B, N, hidden_dim).  The architecture is permutation-equivariant:
    reordering input points reorders the output features identically.

    Args:
        hidden_dim: Feature dimension throughout the network (default 1024).
        num_layers: Number of PVConvBlock layers (default 4).
        voxel_resolution: Side length of the voxel grid (default 32).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        voxel_resolution: int = 32,
    ):
        super().__init__()
        self.input_proj = SharedMLP(3, hidden_dim)
        self.blocks = nn.ModuleList(
            [PVConvBlock(hidden_dim, voxel_resolution) for _ in range(num_layers)]
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) point coordinates (assumed in [-0.5, 0.5])
        Returns:
            (B, N, hidden_dim) per-point features
        """
        features = self.input_proj(coords)  # (B, N, hidden_dim)
        for block in self.blocks:
            residual = features
            features = block(features, coords) + residual  # residual connection
        return features

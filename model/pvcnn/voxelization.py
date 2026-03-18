import torch
import torch.nn as nn


class Voxelization(nn.Module):
    """Voxel branch: scatter point features into a 3-D grid, apply 3-D
    convolution, then query features back at the original point locations.

    Coordinates are assumed to be normalized to [-0.5, 0.5].
    """

    def __init__(self, in_channels: int, out_channels: int, resolution: int = 32):
        super().__init__()
        self.resolution = resolution
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _coords_to_voxel_indices(self, coords: torch.Tensor) -> torch.Tensor:
        """Map continuous coords in [-0.5, 0.5] to integer voxel indices.

        Args:
            coords: (B, N, 3)
        Returns:
            (B, N, 3) long tensor with values in [0, resolution-1]
        """
        r = self.resolution
        # shift to [0, 1] then scale to [0, r-1]
        indices = ((coords + 0.5) * r).clamp(0, r - 1).long()
        return indices

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, C_in)  per-point features
            coords:   (B, N, 3)     point coordinates in [-0.5, 0.5]
        Returns:
            (B, N, C_out)  voxel-processed features queried back at point locations
        """
        B, N, C = features.shape
        r = self.resolution
        device = features.device

        voxel_idx = self._coords_to_voxel_indices(coords)  # (B, N, 3)

        # --- scatter into voxel grid ---
        # flat index for scatter_add
        ix = voxel_idx[..., 0]  # (B, N)
        iy = voxel_idx[..., 1]
        iz = voxel_idx[..., 2]
        flat = ix * (r * r) + iy * r + iz  # (B, N)

        # Accumulate features and counts per voxel
        grid_flat = torch.zeros(B, r * r * r, C, device=device)
        counts = torch.zeros(B, r * r * r, 1, device=device)

        flat_exp = flat.unsqueeze(-1).expand_as(features)  # (B, N, C)
        grid_flat.scatter_add_(1, flat_exp, features)
        counts.scatter_add_(1, flat.unsqueeze(-1), torch.ones(B, N, 1, device=device))

        # Average pooling (avoid /0)
        counts = counts.clamp(min=1)
        grid_flat = grid_flat / counts

        # reshape to (B, C, D, H, W)
        grid = grid_flat.transpose(1, 2).reshape(B, C, r, r, r)

        # --- 3D convolution ---
        grid = self.conv3d(grid)  # (B, C_out, r, r, r)
        C_out = grid.shape[1]

        # --- query back at point locations ---
        grid_flat_out = grid.reshape(B, C_out, r * r * r).transpose(1, 2)  # (B, r^3, C_out)
        flat_exp_out = flat.unsqueeze(-1).expand(B, N, C_out)
        point_features = torch.gather(grid_flat_out, 1, flat_exp_out)  # (B, N, C_out)

        return point_features

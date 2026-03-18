"""Farthest Point Sampling (FPS) and Level-of-Detail (LoD) sequence construction.

Implements FPS for sub-sampling point clouds and builds LoD sequences
fine-to-coarse following Eq. 17: X_{k-1} = FPS(X_k), ensuring the subset
property X_1 ⊂ X_2 ⊂ ... ⊂ X_K.
"""

import torch


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Sample ``num_samples`` points from ``points`` using Farthest Point Sampling.

    Args:
        points: (B, N, 3) point cloud tensor.
        num_samples: number of points to sample.  Must satisfy 1 <= num_samples <= N.

    Returns:
        indices: (B, num_samples) long tensor of selected point indices into the
                 N dimension of ``points``.
    """
    device = points.device
    B, N, C = points.shape
    assert 1 <= num_samples <= N, (
        f"num_samples={num_samples} must be in [1, {N}]"
    )

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]

    return centroids


def build_lod_sequence(
    points: torch.Tensor,
    scale_points: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build a Level-of-Detail sequence fine-to-coarse (Eq. 17).

    Given the full point cloud X_K = ``points`` with N points and a list of
    ascending scale sizes [s_1, s_2, ..., s_K] where s_K = N, this function
    produces the LoD hierarchy by repeatedly applying FPS from the finest
    available scale downward:

        X_{k-1} = FPS(X_k, s_{k-1})

    The subset property is guaranteed: the indices at scale k are always a
    subset of the indices at scale k+1.

    Args:
        points: (B, N, 3) full point cloud.
        scale_points: ascending list of integers, e.g. [4, 16, 64, 256, 2048].
                      The last element must equal N.

    Returns:
        lod_points:  list of K tensors, each (B, s_k, 3).
        lod_indices: list of K tensors, each (B, s_k) long — indices into the
                     *original* N points.
    """
    B, N, C = points.shape
    assert scale_points[-1] == N, (
        f"Last scale ({scale_points[-1]}) must equal N ({N})"
    )
    assert all(
        scale_points[i] < scale_points[i + 1]
        for i in range(len(scale_points) - 1)
    ), "scale_points must be strictly ascending"

    K = len(scale_points)

    # Start from the finest (full) scale.
    # current_indices maps positions in the current working set back to the
    # original point indices.
    current_indices = torch.arange(N, dtype=torch.long, device=points.device)
    current_indices = current_indices.unsqueeze(0).expand(B, -1)  # (B, N)
    current_points = points  # (B, N, 3)

    # We build from fine to coarse, then reverse.
    lod_points: list[torch.Tensor] = [None] * K  # type: ignore[list-item]
    lod_indices: list[torch.Tensor] = [None] * K  # type: ignore[list-item]

    # The finest scale is the full point cloud.
    lod_points[K - 1] = current_points
    lod_indices[K - 1] = current_indices

    for k in range(K - 2, -1, -1):
        num_samples = scale_points[k]
        # FPS on the current working set to get local indices.
        local_idx = farthest_point_sampling(current_points, num_samples)  # (B, num_samples)

        # Map local indices back to the original point cloud indices.
        global_idx = torch.gather(current_indices, 1, local_idx)  # (B, num_samples)

        # Gather the actual 3-D coordinates.
        local_idx_expanded = local_idx.unsqueeze(-1).expand(-1, -1, C)
        sub_points = torch.gather(current_points, 1, local_idx_expanded)  # (B, num_samples, 3)

        lod_points[k] = sub_points
        lod_indices[k] = global_idx

        # The next (coarser) iteration works on the sub-sampled set.
        current_points = sub_points
        current_indices = global_idx

    return lod_points, lod_indices

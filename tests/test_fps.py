"""Tests for FPS and LoD sequence construction."""

import torch
import pytest

from model.fps import farthest_point_sampling, build_lod_sequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_points():
    """Return a deterministic (B=2, N=64, 3) point cloud."""
    torch.manual_seed(42)
    return torch.randn(2, 64, 3)


# ---------------------------------------------------------------------------
# farthest_point_sampling tests
# ---------------------------------------------------------------------------

class TestFarthestPointSampling:
    def test_output_shape(self, sample_points):
        B, N, _ = sample_points.shape
        num_samples = 16
        idx = farthest_point_sampling(sample_points, num_samples)
        assert idx.shape == (B, num_samples)

    def test_indices_dtype_is_long(self, sample_points):
        idx = farthest_point_sampling(sample_points, 8)
        assert idx.dtype == torch.long

    def test_indices_unique_per_batch(self, sample_points):
        """Each batch element should have no duplicate indices."""
        idx = farthest_point_sampling(sample_points, 32)
        for b in range(idx.shape[0]):
            unique = torch.unique(idx[b])
            assert unique.numel() == idx.shape[1], (
                f"Batch {b}: expected {idx.shape[1]} unique indices, got {unique.numel()}"
            )

    def test_indices_in_valid_range(self, sample_points):
        B, N, _ = sample_points.shape
        idx = farthest_point_sampling(sample_points, 16)
        assert (idx >= 0).all() and (idx < N).all()

    def test_subsampled_points_are_subset(self, sample_points):
        """Points gathered via FPS indices must exist in the original cloud."""
        idx = farthest_point_sampling(sample_points, 16)
        gathered = torch.gather(
            sample_points, 1, idx.unsqueeze(-1).expand(-1, -1, 3)
        )
        for b in range(sample_points.shape[0]):
            for i in range(idx.shape[1]):
                original_pt = sample_points[b, idx[b, i]]
                assert torch.allclose(gathered[b, i], original_pt)

    def test_sample_all_points(self, sample_points):
        """Sampling N out of N should return all indices (in some order)."""
        B, N, _ = sample_points.shape
        idx = farthest_point_sampling(sample_points, N)
        for b in range(B):
            assert set(idx[b].tolist()) == set(range(N))


# ---------------------------------------------------------------------------
# build_lod_sequence tests
# ---------------------------------------------------------------------------

class TestBuildLodSequence:
    def test_output_shapes(self):
        torch.manual_seed(0)
        B, N = 2, 128
        points = torch.randn(B, N, 3)
        scale_points = [4, 16, 64, 128]

        lod_pts, lod_idx = build_lod_sequence(points, scale_points)

        assert len(lod_pts) == len(scale_points)
        assert len(lod_idx) == len(scale_points)
        for k, s in enumerate(scale_points):
            assert lod_pts[k].shape == (B, s, 3), f"Scale {k}: wrong point shape"
            assert lod_idx[k].shape == (B, s), f"Scale {k}: wrong index shape"

    def test_finest_scale_is_full_cloud(self):
        torch.manual_seed(0)
        B, N = 2, 64
        points = torch.randn(B, N, 3)
        scale_points = [8, 32, 64]

        lod_pts, lod_idx = build_lod_sequence(points, scale_points)

        # Finest scale should be the original point cloud.
        assert torch.allclose(lod_pts[-1], points)
        expected_idx = torch.arange(N).unsqueeze(0).expand(B, -1)
        assert (lod_idx[-1] == expected_idx).all()

    def test_subset_property(self):
        """Indices at coarser scale k must be a subset of indices at scale k+1."""
        torch.manual_seed(7)
        B, N = 3, 256
        points = torch.randn(B, N, 3)
        scale_points = [4, 16, 64, 256]

        _, lod_idx = build_lod_sequence(points, scale_points)

        for b in range(B):
            for k in range(len(scale_points) - 1):
                coarse = set(lod_idx[k][b].tolist())
                fine = set(lod_idx[k + 1][b].tolist())
                assert coarse.issubset(fine), (
                    f"Batch {b}, scale {k}: coarse indices are not a subset of "
                    f"finer scale {k+1}. Missing: {coarse - fine}"
                )

    def test_points_match_indices(self):
        """lod_points[k] must equal points gathered at lod_indices[k]."""
        torch.manual_seed(1)
        B, N = 2, 128
        points = torch.randn(B, N, 3)
        scale_points = [8, 32, 128]

        lod_pts, lod_idx = build_lod_sequence(points, scale_points)

        for k, s in enumerate(scale_points):
            gathered = torch.gather(
                points, 1, lod_idx[k].unsqueeze(-1).expand(-1, -1, 3)
            )
            assert torch.allclose(lod_pts[k], gathered), (
                f"Scale {k}: lod_points do not match points at lod_indices"
            )

    def test_last_scale_must_equal_n(self):
        points = torch.randn(2, 64, 3)
        with pytest.raises(AssertionError):
            build_lod_sequence(points, [8, 32])

    def test_scales_must_be_ascending(self):
        points = torch.randn(2, 64, 3)
        with pytest.raises(AssertionError):
            build_lod_sequence(points, [32, 16, 64])

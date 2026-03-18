import torch
import pytest

from model.pvcnn import PVCNN


@pytest.fixture
def default_model():
    """PVCNN with small hidden_dim for fast testing."""
    return PVCNN(hidden_dim=64, num_layers=2, voxel_resolution=16)


class TestPVCNNOutputShape:
    def test_basic_shape(self, default_model):
        B, N, hidden_dim = 2, 256, 64
        coords = torch.rand(B, N, 3) - 0.5  # [-0.5, 0.5]
        out = default_model(coords)
        assert out.shape == (B, N, hidden_dim)

    def test_single_batch(self, default_model):
        coords = torch.rand(1, 128, 3) - 0.5
        out = default_model(coords)
        assert out.shape == (1, 128, 64)

    def test_paper_config(self):
        """Full paper config: hidden_dim=1024, 4 layers, voxel_resolution=32."""
        model = PVCNN(hidden_dim=1024, num_layers=4, voxel_resolution=32)
        coords = torch.rand(1, 64, 3) - 0.5
        out = model(coords)
        assert out.shape == (1, 64, 1024)


class TestPVCNNPermutationEquivariance:
    def test_permutation_equivariance(self, default_model):
        """pi(PVCNN(x)) == PVCNN(pi(x))"""
        B, N = 2, 256
        coords = torch.rand(B, N, 3) - 0.5

        # Random permutation (same permutation for both batches for simplicity)
        perm = torch.randperm(N)

        default_model.eval()
        with torch.no_grad():
            out_original = default_model(coords)            # (B, N, C)
            out_permuted_input = default_model(coords[:, perm])  # (B, N, C)

        # Permuting the output of the original should match the output
        # when the input was permuted.
        out_original_permuted = out_original[:, perm]

        torch.testing.assert_close(
            out_original_permuted,
            out_permuted_input,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_permutation_equivariance_different_perms(self, default_model):
        """Test with a second random permutation to reduce fluke chance."""
        B, N = 1, 128
        coords = torch.rand(B, N, 3) - 0.5
        perm = torch.randperm(N)

        default_model.eval()
        with torch.no_grad():
            out_orig = default_model(coords)
            out_perm = default_model(coords[:, perm])

        torch.testing.assert_close(
            out_orig[:, perm],
            out_perm,
            atol=1e-4,
            rtol=1e-4,
        )

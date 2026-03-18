import pytest
import torch

from model.upsampling import punet_upsample


class TestPunetUpsample:
    """Tests for PU-Net style upsampling (Eq. 6)."""

    @pytest.mark.parametrize(
        "B, s_k, d, target_n",
        [
            (2, 64, 128, 256),
            (1, 32, 64, 128),
            (4, 16, 32, 64),
            (2, 100, 256, 1000),
            (1, 1, 16, 8),
        ],
    )
    def test_output_shape(self, B, s_k, d, target_n):
        z = torch.randn(B, s_k, d)
        z_up = punet_upsample(z, target_n)
        assert z_up.shape == (B, target_n, d)

    def test_noop_when_equal(self):
        z = torch.randn(2, 64, 128)
        z_up = punet_upsample(z, 64)
        assert z_up is z  # should return the exact same tensor

    def test_duplicated_blocks_identical(self):
        B, s_k, d, target_n = 2, 4, 8, 16
        r = target_n // s_k  # 4
        z = torch.randn(B, s_k, d)
        z_up = punet_upsample(z, target_n)
        # Each group of r consecutive rows should be identical
        for i in range(s_k):
            block = z_up[:, i * r : (i + 1) * r, :]
            for j in range(r):
                assert torch.equal(block[:, j, :], z[:, i, :])

    def test_assertion_error_not_divisible(self):
        z = torch.randn(2, 3, 16)
        with pytest.raises(AssertionError, match="must be divisible"):
            punet_upsample(z, 10)

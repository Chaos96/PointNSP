import torch
import pytest

from model.masking import build_blockwise_causal_mask, build_position_aware_soft_mask


# ---------------------------------------------------------------------------
# Block-wise causal mask tests
# ---------------------------------------------------------------------------

class TestBlockwiseCausalMask:
    def test_output_shape(self):
        """Output shape is (T, T) where T = sum(scale_points)."""
        scale_points = [4, 8, 16]
        mask = build_blockwise_causal_mask(scale_points)
        T = sum(scale_points)
        assert mask.shape == (T, T)

    def test_two_scales(self):
        """Basic two-scale test: scale 1 sees only itself, scale 2 sees both."""
        scale_points = [4, 8]
        mask = build_blockwise_causal_mask(scale_points)

        # Intra-scale block 1 (top-left 4x4) all 1s
        assert mask[:4, :4].equal(torch.ones(4, 4))
        # Intra-scale block 2 (bottom-right 8x8) all 1s
        assert mask[4:12, 4:12].equal(torch.ones(8, 8))
        # Cross-scale lower triangle: scale 2 attends to scale 1
        assert mask[4:12, :4].equal(torch.ones(8, 4))
        # Cross-scale upper triangle: scale 1 cannot attend to scale 2
        assert mask[:4, 4:12].equal(torch.zeros(4, 8))

    def test_three_scales_full_causal_structure(self):
        """Three scales: verify full block-causal structure."""
        scale_points = [2, 3, 5]
        mask = build_blockwise_causal_mask(scale_points)
        offsets = [0, 2, 5, 10]

        K = len(scale_points)
        for i in range(K):
            for j in range(K):
                block = mask[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
                si, sj = scale_points[i], scale_points[j]
                if j <= i:
                    # Should be all 1s (attend)
                    assert block.equal(torch.ones(si, sj)), \
                        f"Block ({i},{j}) should be all 1s"
                else:
                    # Should be all 0s (masked)
                    assert block.equal(torch.zeros(si, sj)), \
                        f"Block ({i},{j}) should be all 0s"

    def test_diagonal_blocks_are_ones(self):
        """All diagonal blocks (intra-scale) must be all 1s."""
        scale_points = [3, 7, 4, 6]
        mask = build_blockwise_causal_mask(scale_points)
        offset = 0
        for s in scale_points:
            block = mask[offset:offset+s, offset:offset+s]
            assert block.equal(torch.ones(s, s))
            offset += s

    def test_upper_triangle_blocks_are_zeros(self):
        """All strictly upper-triangular blocks must be all 0s."""
        scale_points = [3, 5, 2]
        mask = build_blockwise_causal_mask(scale_points)
        offsets = [0, 3, 8, 10]

        K = len(scale_points)
        for i in range(K):
            for j in range(i + 1, K):
                block = mask[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
                assert block.equal(torch.zeros(scale_points[i], scale_points[j]))

    def test_single_scale(self):
        """Single scale: entire mask should be all 1s."""
        mask = build_blockwise_causal_mask([10])
        assert mask.equal(torch.ones(10, 10))

    def test_device(self):
        """Mask should be created on the specified device."""
        mask = build_blockwise_causal_mask([4, 8], device=torch.device("cpu"))
        assert mask.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Position-aware soft mask tests
# ---------------------------------------------------------------------------

class TestPositionAwareSoftMask:
    def test_output_shape(self):
        """Output shape is (B, s_k, s_k)."""
        B, s_k, d = 2, 16, 64
        P_k = torch.randn(B, s_k, d)
        W_p = torch.randn(d, d)
        mask = build_position_aware_soft_mask(P_k, W_p)
        assert mask.shape == (B, s_k, s_k)

    def test_values_in_range(self):
        """All values must be in [0, 1]."""
        B, s_k, d = 4, 8, 32
        P_k = torch.randn(B, s_k, d)
        W_p = torch.randn(d, d)
        mask = build_position_aware_soft_mask(P_k, W_p)
        assert (mask >= 0).all()
        assert (mask <= 1).all()

    def test_rows_sum_to_one(self):
        """Each row should sum to 1 (softmax property)."""
        B, s_k, d = 3, 10, 16
        P_k = torch.randn(B, s_k, d)
        W_p = torch.randn(d, d)
        mask = build_position_aware_soft_mask(P_k, W_p)
        row_sums = mask.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_different_coordinates_produce_different_masks(self):
        """Different positional embeddings should yield different masks."""
        B, s_k, d = 1, 8, 16
        # Use small-scale values so softmax doesn't saturate to near-identity
        W_p = torch.randn(d, d) * 0.1
        P_k_a = torch.randn(B, s_k, d) * 0.1
        P_k_b = torch.randn(B, s_k, d) * 0.1
        mask_a = build_position_aware_soft_mask(P_k_a, W_p)
        mask_b = build_position_aware_soft_mask(P_k_b, W_p)
        assert not torch.allclose(mask_a, mask_b, atol=1e-6)

    def test_batch_independence(self):
        """Each batch element should be computed independently."""
        B, s_k, d = 2, 6, 16
        P_k = torch.randn(B, s_k, d)
        W_p = torch.randn(d, d)
        mask_full = build_position_aware_soft_mask(P_k, W_p)
        mask_0 = build_position_aware_soft_mask(P_k[0:1], W_p)
        mask_1 = build_position_aware_soft_mask(P_k[1:2], W_p)
        assert torch.allclose(mask_full[0], mask_0[0], atol=1e-6)
        assert torch.allclose(mask_full[1], mask_1[0], atol=1e-6)

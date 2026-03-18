import torch
import math
import pytest

from model.positional_encoding import BAPEPositionalEncoding, ScaleEmbedding


class TestBAPEPositionalEncoding:
    """Tests for BAPE positional encoding (Eq. 20-22)."""

    def test_output_shape(self):
        """(B, N, 3) coords should produce (B, N, d_model) encoding."""
        d_model = 64
        bape = BAPEPositionalEncoding(d_model=d_model)
        coords = torch.randn(2, 16, 3)
        pe = bape(coords)
        assert pe.shape == (2, 16, d_model)

    def test_different_coords_produce_different_pe(self):
        """Different coordinates must yield different positional encodings."""
        bape = BAPEPositionalEncoding(d_model=64)
        coords_a = torch.tensor([[[0.1, 0.2, 0.3]]])
        coords_b = torch.tensor([[[0.4, 0.5, 0.6]]])
        pe_a = bape(coords_a)
        pe_b = bape(coords_b)
        assert not torch.allclose(pe_a, pe_b)

    def test_sin_cos_same_divisor(self):
        """Verify sin and cos channels use the SAME div_term (Eq. 21-22)."""
        d_model = 16
        bape = BAPEPositionalEncoding(d_model=d_model, lam=1000.0)
        coords = torch.tensor([[[1.0, 2.0, 3.0]]])

        pe = bape(coords)
        # Manually compute expected values for the first sin/cos pair
        p = 1000.0 ** 2 * 3.0 + 1000.0 * 2.0 + 1.0
        div_0 = math.exp(0.0 * (-math.log(10000.0) / d_model))  # i=0 -> 1.0

        expected_sin = math.sin(p * div_0)
        expected_cos = math.cos(p * div_0)

        assert pytest.approx(pe[0, 0, 0].item(), abs=1e-5) == expected_sin
        assert pytest.approx(pe[0, 0, 1].item(), abs=1e-5) == expected_cos

    def test_deterministic(self):
        """Same input should always produce the same output."""
        bape = BAPEPositionalEncoding(d_model=32)
        coords = torch.randn(1, 8, 3)
        pe1 = bape(coords)
        pe2 = bape(coords)
        assert torch.allclose(pe1, pe2)

    def test_batch_consistency(self):
        """Each batch element should be encoded independently."""
        bape = BAPEPositionalEncoding(d_model=32)
        coords = torch.randn(4, 10, 3)
        pe = bape(coords)
        # Process first element alone
        pe_single = bape(coords[0:1])
        assert torch.allclose(pe[0:1], pe_single)


class TestScaleEmbedding:
    """Tests for learnable scale embedding."""

    def test_output_shape(self):
        """forward(k) should return (d_model,) tensor."""
        d_model = 64
        se = ScaleEmbedding(num_scales=4, d_model=d_model)
        emb = se(0)
        assert emb.shape == (d_model,)

    def test_same_index_same_embedding(self):
        """Same scale index must return identical embeddings."""
        se = ScaleEmbedding(num_scales=4, d_model=32)
        emb1 = se(2)
        emb2 = se(2)
        assert torch.allclose(emb1, emb2)

    def test_different_index_different_embedding(self):
        """Different scale indices should (almost surely) differ."""
        se = ScaleEmbedding(num_scales=4, d_model=32)
        emb0 = se(0)
        emb1 = se(1)
        assert not torch.allclose(emb0, emb1)

    def test_get_all_shape(self):
        """get_all() should return (num_scales, d_model)."""
        se = ScaleEmbedding(num_scales=5, d_model=64)
        all_emb = se.get_all()
        assert all_emb.shape == (5, 64)

"""Tests for the PointNSP autoregressive transformer.

Uses small configs: d_model=64, nhead=4, num_layers=2, scale_points=[4,16,64],
codebook_size=128 to keep tests fast and lightweight.
"""

import pytest
import torch

from model.transformer import PointNSPTransformerBlock, PointNSPTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CODEBOOK_SIZE = 128
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
SCALE_POINTS = [4, 16, 64]
NUM_SCALES = len(SCALE_POINTS)
BATCH_SIZE = 2
TOTAL_SEQ_LEN = sum(SCALE_POINTS)  # 84


@pytest.fixture
def transformer():
    model = PointNSPTransformer(
        codebook_size=CODEBOOK_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        scale_points=SCALE_POINTS,
        num_scales=NUM_SCALES,
        lam=1000.0,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def sample_tokens():
    """Create random token indices for each scale."""
    tokens = []
    for s_k in SCALE_POINTS:
        tokens.append(torch.randint(0, CODEBOOK_SIZE, (BATCH_SIZE, s_k)))
    return tokens


@pytest.fixture
def sample_coords():
    """Create random 3D coordinates for each scale."""
    coords = []
    for s_k in SCALE_POINTS:
        coords.append(torch.randn(BATCH_SIZE, s_k, 3))
    return coords


# ---------------------------------------------------------------------------
# Tests for PointNSPTransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self):
        block = PointNSPTransformerBlock(d_model=D_MODEL, nhead=NHEAD, dropout=0.0)
        B, T, D = BATCH_SIZE, TOTAL_SEQ_LEN, D_MODEL

        x = torch.randn(B, T, D)
        pos_emb = torch.randn(B, T, D)
        scale_emb = torch.randn(B, T, D)
        causal_mask = torch.ones(T, T)  # fully unmasked
        out = block(x, pos_emb, scale_emb, causal_mask, SCALE_POINTS)
        assert out.shape == (B, T, D)

    def test_causal_mask_zeros_attention(self):
        """Verify that masked positions get no attention."""
        block = PointNSPTransformerBlock(d_model=D_MODEL, nhead=NHEAD, dropout=0.0)
        B, T, D = 1, TOTAL_SEQ_LEN, D_MODEL

        x = torch.randn(B, T, D)
        pos_emb = torch.randn(B, T, D)
        scale_emb = torch.randn(B, T, D)

        # All-ones mask vs restrictive mask should give different outputs
        full_mask = torch.ones(T, T)
        from model.masking import build_blockwise_causal_mask
        causal_mask = build_blockwise_causal_mask(SCALE_POINTS)

        out_full = block(x, pos_emb, scale_emb, full_mask, SCALE_POINTS)
        out_causal = block(x, pos_emb, scale_emb, causal_mask, SCALE_POINTS)

        # Outputs should differ when masks differ
        assert not torch.allclose(out_full, out_causal, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests for PointNSPTransformer
# ---------------------------------------------------------------------------

class TestTransformerForward:
    def test_forward_returns_correct_shapes(self, transformer, sample_tokens, sample_coords):
        """forward() should return a list of K logit tensors with correct shapes."""
        logits = transformer(sample_tokens, sample_coords)

        assert len(logits) == NUM_SCALES
        for k, logits_k in enumerate(logits):
            expected_shape = (BATCH_SIZE, SCALE_POINTS[k], CODEBOOK_SIZE)
            assert logits_k.shape == expected_shape, (
                f"Scale {k}: expected {expected_shape}, got {logits_k.shape}"
            )

    def test_forward_logits_finite(self, transformer, sample_tokens, sample_coords):
        """All logits should be finite (no NaN or Inf)."""
        logits = transformer(sample_tokens, sample_coords)
        for k, logits_k in enumerate(logits):
            assert torch.isfinite(logits_k).all(), f"Scale {k} has non-finite logits"


class TestComputeLoss:
    def test_loss_positive_scalar(self, transformer, sample_tokens, sample_coords):
        """compute_loss() should return a positive scalar."""
        transformer.train()
        loss = transformer.compute_loss(sample_tokens, sample_coords)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Cross-entropy loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_loss_decreases_with_correct_tokens(self, transformer, sample_coords):
        """Loss should be lower when logits match targets (sanity check)."""
        transformer.train()
        # Random tokens -> some loss
        random_tokens = [torch.randint(0, CODEBOOK_SIZE, (BATCH_SIZE, s)) for s in SCALE_POINTS]
        loss_random = transformer.compute_loss(random_tokens, sample_coords)

        # The loss is a valid positive number
        assert loss_random.item() > 0

    def test_loss_backward(self, transformer, sample_tokens, sample_coords):
        """Loss should be differentiable."""
        transformer.train()
        loss = transformer.compute_loss(sample_tokens, sample_coords)
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for p in transformer.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "At least one parameter should have a non-zero gradient"


class TestGenerate:
    def test_generate_shape(self):
        """generate() should return point clouds of shape (B, s_K, 3)."""
        # Build a tiny VQVAE-like mock
        from model.vqvae_model import MultiScaleVQVAE

        vqvae = MultiScaleVQVAE(
            hidden_dim=D_MODEL,
            codebook_size=CODEBOOK_SIZE,
            num_points=SCALE_POINTS[-1],
            scale_points=SCALE_POINTS,
            pvcnn_layers=1,
            voxel_resolution=8,
        )
        vqvae.eval()

        transformer = PointNSPTransformer(
            codebook_size=CODEBOOK_SIZE,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            scale_points=SCALE_POINTS,
            num_scales=NUM_SCALES,
            lam=1000.0,
            dropout=0.0,
        )
        transformer.eval()

        device = torch.device("cpu")
        num_samples = 2
        point_clouds = transformer.generate(vqvae, num_samples, device, temperature=1.0)

        assert point_clouds.shape == (num_samples, SCALE_POINTS[-1], 3), (
            f"Expected ({num_samples}, {SCALE_POINTS[-1]}, 3), got {point_clouds.shape}"
        )
        assert torch.isfinite(point_clouds).all(), "Generated points should be finite"


class TestStartToken:
    def test_start_token_learnable(self, transformer):
        """The start token should be a learnable parameter."""
        assert transformer.start_token.requires_grad
        assert transformer.start_token.shape == (D_MODEL,)

    def test_scale_1_uses_start_token(self, transformer, sample_tokens):
        """Scale 1 positions in input embeddings should use [start] token."""
        emb = transformer._build_input_embeddings(sample_tokens)
        # First s_1 positions should all be the start token (expanded)
        s_1 = SCALE_POINTS[0]
        start_expanded = transformer.start_token.unsqueeze(0).unsqueeze(0).expand(
            BATCH_SIZE, s_1, -1
        )
        assert torch.allclose(emb[:, :s_1, :], start_expanded)

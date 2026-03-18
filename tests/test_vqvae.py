"""Tests for Multi-Scale VQVAE model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from model.vqvae_model import (
    MultiScaleVQVAE,
    VectorQuantizer,
    PhiNetwork,
    chamfer_distance,
    earth_movers_distance_approx,
)

# Small test config
TEST_HIDDEN_DIM = 64
TEST_CODEBOOK_SIZE = 128
TEST_NUM_POINTS = 64
TEST_SCALE_POINTS = [4, 16, 64]
TEST_PVCNN_LAYERS = 1
TEST_VOXEL_RESOLUTION = 8
TEST_BATCH_SIZE = 2


@pytest.fixture
def model():
    m = MultiScaleVQVAE(
        hidden_dim=TEST_HIDDEN_DIM,
        codebook_size=TEST_CODEBOOK_SIZE,
        num_points=TEST_NUM_POINTS,
        scale_points=TEST_SCALE_POINTS,
        pvcnn_layers=TEST_PVCNN_LAYERS,
        voxel_resolution=TEST_VOXEL_RESOLUTION,
    )
    m.eval()
    return m


@pytest.fixture
def sample_input():
    return torch.randn(TEST_BATCH_SIZE, TEST_NUM_POINTS, 3) * 0.5


class TestForward:
    def test_forward_returns_correct_shapes(self, model, sample_input):
        x_hat, total_loss, recon_loss, vq_loss = model(sample_input)
        assert x_hat.shape == (TEST_BATCH_SIZE, TEST_NUM_POINTS, 3)

    def test_forward_losses_are_positive_scalars(self, model, sample_input):
        _, total_loss, recon_loss, vq_loss = model(sample_input)
        assert total_loss.dim() == 0, "total_loss should be a scalar"
        assert recon_loss.dim() == 0, "recon_loss should be a scalar"
        assert vq_loss.dim() == 0, "vq_loss should be a scalar"
        assert total_loss.item() > 0, "total_loss should be positive"
        assert recon_loss.item() > 0, "recon_loss should be positive"
        assert vq_loss.item() > 0, "vq_loss should be positive"


class TestEncode:
    def test_encode_produces_correct_number_of_token_lists(self, model, sample_input):
        token_lists, vq_losses, _ = model.encode(sample_input)
        assert len(token_lists) == len(TEST_SCALE_POINTS)
        assert len(vq_losses) == len(TEST_SCALE_POINTS)

    def test_encode_token_shapes(self, model, sample_input):
        token_lists, _, _ = model.encode(sample_input)
        for k, tokens in enumerate(token_lists):
            expected_n = TEST_SCALE_POINTS[k]
            assert tokens.shape == (TEST_BATCH_SIZE, expected_n), (
                f"Scale {k}: expected ({TEST_BATCH_SIZE}, {expected_n}), "
                f"got {tokens.shape}"
            )
            assert tokens.dtype == torch.long

    def test_encode_vq_losses_are_positive_scalars(self, model, sample_input):
        _, vq_losses, _ = model.encode(sample_input)
        for k, loss in enumerate(vq_losses):
            assert loss.dim() == 0, f"vq_loss at scale {k} should be a scalar"
            assert loss.item() > 0, f"vq_loss at scale {k} should be positive"


class TestDecode:
    def test_decode_from_tokens_produces_correct_shape(self, model, sample_input):
        token_lists, _, _ = model.encode(sample_input)
        x_hat = model.decode(token_lists)
        assert x_hat.shape == (TEST_BATCH_SIZE, TEST_NUM_POINTS, 3)


class TestVectorQuantizer:
    def test_vq_output_shapes(self):
        vq = VectorQuantizer(codebook_size=128, embedding_dim=64, num_scales=3)
        z = torch.randn(2, 16, 64)
        quantized, vq_loss, indices = vq(z)
        assert quantized.shape == z.shape
        assert indices.shape == (2, 16)
        assert indices.dtype == torch.long
        assert vq_loss.dim() == 0


class TestPhiNetwork:
    def test_phi_preserves_shape(self):
        phi = PhiNetwork(dim=64, ratio=0.5)
        x = torch.randn(2, 32, 64)
        out = phi(x)
        assert out.shape == x.shape


class TestLosses:
    def test_chamfer_distance_positive(self):
        x = torch.randn(2, 64, 3)
        y = torch.randn(2, 64, 3)
        cd = chamfer_distance(x, y)
        assert cd.dim() == 0
        assert cd.item() > 0

    def test_earth_movers_distance_positive(self):
        x = torch.randn(2, 64, 3)
        y = torch.randn(2, 64, 3)
        emd = earth_movers_distance_approx(x, y, n_iters=10)
        assert emd.dim() == 0
        assert emd.item() > 0

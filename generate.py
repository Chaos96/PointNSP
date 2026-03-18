"""Generation script for PointNSP.

Loads pretrained VQVAE and Transformer checkpoints, generates point cloud
samples autoregressively, and saves them as .npy files.

Usage:
    python generate.py \
        --config configs/transformer_medium.yaml \
        --vqvae_ckpt checkpoints/vqvae_best.pt \
        --transformer_ckpt checkpoints/transformer_best.pt \
        --num_samples 16 \
        --output_dir generated/
"""

import argparse
import os

import numpy as np
import torch
import yaml

from model.vqvae_model import MultiScaleVQVAE
from model.transformer import PointNSPTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate point clouds with PointNSP")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to transformer YAML config"
    )
    parser.add_argument(
        "--vqvae_ckpt", type=str, required=True, help="Path to pretrained VQVAE checkpoint"
    )
    parser.add_argument(
        "--vqvae_config", type=str, default=None,
        help="Path to VQVAE config (if not stored in transformer config)"
    )
    parser.add_argument(
        "--transformer_ckpt", type=str, required=True,
        help="Path to pretrained Transformer checkpoint"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of point clouds to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Generation batch size"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated", help="Directory to save outputs"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    vqvae_cfg = cfg.get("vqvae", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- Load VQVAE ----
    if args.vqvae_config:
        vqvae_model_cfg = load_config(args.vqvae_config)["model"]
    else:
        vqvae_model_cfg = vqvae_cfg

    vqvae = MultiScaleVQVAE(
        hidden_dim=vqvae_model_cfg.get("hidden_dim", 1024),
        codebook_size=vqvae_model_cfg.get("codebook_size", 8192),
        num_points=vqvae_model_cfg.get("num_points", 2048),
        scale_points=vqvae_model_cfg.get("scale_points", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        pvcnn_layers=vqvae_model_cfg.get("pvcnn_layers", 4),
        voxel_resolution=vqvae_model_cfg.get("voxel_resolution", 32),
    ).to(device)

    ckpt = torch.load(args.vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VQVAE from {args.vqvae_ckpt}")

    # ---- Load Transformer ----
    scale_points = model_cfg["scale_points"]
    transformer = PointNSPTransformer(
        codebook_size=model_cfg["codebook_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        scale_points=scale_points,
        num_scales=len(scale_points),
        lam=model_cfg.get("lam", 1000.0),
        dropout=0.0,  # no dropout at inference
    ).to(device)

    ckpt = torch.load(args.transformer_ckpt, map_location=device)
    transformer.load_state_dict(ckpt["model_state_dict"])
    transformer.eval()
    print(f"Loaded Transformer from {args.transformer_ckpt}")

    # ---- Generate ----
    all_samples = []
    remaining = args.num_samples
    batch_idx = 0

    while remaining > 0:
        bs = min(args.batch_size, remaining)
        print(f"Generating batch {batch_idx + 1} ({bs} samples)...")

        point_clouds = transformer.generate(
            vqvae=vqvae,
            num_samples=bs,
            device=device,
            temperature=args.temperature,
        )
        all_samples.append(point_clouds.cpu().numpy())
        remaining -= bs
        batch_idx += 1

    all_samples = np.concatenate(all_samples, axis=0)  # (num_samples, N, 3)
    print(f"Generated {all_samples.shape[0]} point clouds of shape {all_samples.shape[1:]}")

    # Save
    output_path = os.path.join(args.output_dir, "generated_samples.npy")
    np.save(output_path, all_samples)
    print(f"Saved to {output_path}")

    # Also save individual samples
    for i in range(all_samples.shape[0]):
        np.save(
            os.path.join(args.output_dir, f"sample_{i:04d}.npy"),
            all_samples[i],
        )
    print(f"Saved {all_samples.shape[0]} individual samples to {args.output_dir}/")


if __name__ == "__main__":
    main()

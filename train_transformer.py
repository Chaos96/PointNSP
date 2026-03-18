"""Training script for Autoregressive Transformer (Stage 2).

Loads a pretrained, frozen VQVAE to tokenize point clouds, then trains the
PointNSP transformer with cross-entropy loss on next-scale prediction.

Usage:
    python train_transformer.py --config configs/transformer_medium.yaml --vqvae_ckpt checkpoints/vqvae_best.pt
"""

import argparse
import os
import time

import torch
import torch.optim as optim
import yaml

from model.vqvae_model import MultiScaleVQVAE
from model.transformer import PointNSPTransformer
from model.fps import build_lod_sequence
from datasets.data_processing import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNSP Transformer (Stage 2)")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to transformer YAML config file"
    )
    parser.add_argument(
        "--vqvae_ckpt", type=str, required=True, help="Path to pretrained VQVAE checkpoint"
    )
    parser.add_argument(
        "--vqvae_config", type=str, default=None,
        help="Path to VQVAE config (if not stored in checkpoint)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to transformer checkpoint to resume"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def tokenize_batch(vqvae, x):
    """Tokenize point clouds with frozen VQVAE.

    FPS is stochastic, so different tokens are produced each time (data augmentation).

    Args:
        vqvae: Frozen MultiScaleVQVAE.
        x: (B, N, 3) input point clouds.
    Returns:
        tokens: list of K tensors, each (B, s_k) long indices.
        lod_points: list of K tensors, each (B, s_k, 3) coordinates.
    """
    vqvae.eval()
    tokens, vq_losses, f_res = vqvae.encode(x)

    # Rebuild LoD to get coordinates for BAPE
    _, lod_indices = build_lod_sequence(x, vqvae.scale_points)
    lod_points = []
    for k in range(len(vqvae.scale_points)):
        idx = lod_indices[k].unsqueeze(-1).expand(-1, -1, 3)
        lod_points.append(torch.gather(x, 1, idx))

    return tokens, lod_points


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    vqvae_cfg = cfg.get("vqvae", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load VQVAE (frozen) ----
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
    print(f"Loaded frozen VQVAE from {args.vqvae_ckpt}")

    # ---- Transformer ----
    scale_points = model_cfg["scale_points"]
    transformer = PointNSPTransformer(
        codebook_size=model_cfg["codebook_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        scale_points=scale_points,
        num_scales=len(scale_points),
        lam=model_cfg.get("lam", 1000.0),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"Transformer parameters: {num_params:,}")

    # ---- Dataset ----
    num_points = train_cfg.get("num_points", scale_points[-1])
    categories = data_cfg["categories"]
    if isinstance(categories, list) and len(categories) == 1:
        categories = categories[0]

    train_dataset, test_dataset = get_dataset(
        dataroot=data_cfg["root_dir"],
        npoints=num_points,
        category=categories,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        drop_last=True,
    )

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(
        transformer.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["num_epochs"],
    )

    # ---- Resume ----
    start_epoch = 0
    best_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        transformer.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training loop ----
    num_epochs = train_cfg["num_epochs"]
    save_every = train_cfg.get("save_every", 100)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    for epoch in range(start_epoch, num_epochs):
        transformer.train()
        loss_accum = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in dataloader:
            x = batch["train_points"].to(device)

            # Tokenize with frozen VQVAE (stochastic FPS each time)
            tokens, lod_points = tokenize_batch(vqvae, x)

            optimizer.zero_grad()
            loss = transformer.compute_loss(tokens, lod_points)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=grad_clip)
            optimizer.step()

            loss_accum += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = loss_accum / max(num_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]  "
            f"CE Loss: {avg_loss:.6f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}  "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Checkpoint saving ----
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
            "config": cfg,
        }

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_data["best_loss"] = best_loss
            torch.save(ckpt_data, os.path.join(args.output_dir, "transformer_best.pt"))
            print(f"  -> Saved best checkpoint (loss={best_loss:.6f})")

        if (epoch + 1) % save_every == 0:
            torch.save(
                ckpt_data,
                os.path.join(args.output_dir, f"transformer_epoch{epoch + 1}.pt"),
            )
            print(f"  -> Saved periodic checkpoint at epoch {epoch + 1}")

    print("Training complete.")


if __name__ == "__main__":
    main()

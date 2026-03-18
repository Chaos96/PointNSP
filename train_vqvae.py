"""Training script for Multi-Scale VQVAE (Stage 1).

Usage:
    python train_vqvae.py --config configs/vqvae_medium.yaml
"""

import argparse
import os
import time

import torch
import torch.optim as optim
import yaml

from model.vqvae_model import MultiScaleVQVAE
from datasets.data_processing import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Scale VQVAE")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    model = MultiScaleVQVAE(
        hidden_dim=model_cfg["hidden_dim"],
        codebook_size=model_cfg["codebook_size"],
        num_points=model_cfg["num_points"],
        scale_points=model_cfg["scale_points"],
        pvcnn_layers=model_cfg.get("pvcnn_layers", 4),
        voxel_resolution=model_cfg.get("voxel_resolution", 32),
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Dataset ----
    num_points = train_cfg.get("num_points", model_cfg["num_points"])
    categories = data_cfg["categories"]
    # get_dataset expects a single category string
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
        model.parameters(),
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
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training loop ----
    num_epochs = train_cfg["num_epochs"]
    save_every = train_cfg.get("save_every", 100)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss_accum = 0.0
        total_recon_accum = 0.0
        total_vq_accum = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in dataloader:
            x = batch["train_points"].to(device)

            optimizer.zero_grad()
            x_hat, total_loss, recon_loss, vq_loss = model(x)
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss_accum += total_loss.item()
            total_recon_accum += recon_loss.item()
            total_vq_accum += vq_loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss_accum / max(num_batches, 1)
        avg_recon = total_recon_accum / max(num_batches, 1)
        avg_vq = total_vq_accum / max(num_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]  "
            f"Loss: {avg_loss:.6f}  Recon: {avg_recon:.6f}  VQ: {avg_vq:.6f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}  Time: {elapsed:.1f}s"
        )

        # ---- Checkpoint saving ----
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
        }

        # Best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_data["best_loss"] = best_loss
            torch.save(ckpt_data, os.path.join(args.output_dir, "vqvae_best.pt"))
            print(f"  -> Saved best checkpoint (loss={best_loss:.6f})")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save(
                ckpt_data,
                os.path.join(args.output_dir, f"vqvae_epoch{epoch + 1}.pt"),
            )
            print(f"  -> Saved periodic checkpoint at epoch {epoch + 1}")

    print("Training complete.")


if __name__ == "__main__":
    main()

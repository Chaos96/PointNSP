"""Fast transformer training from pre-tokenized data.

Usage:
    python train_transformer_fast.py --config configs/transformer_small.yaml \
        --tokenized_data data/tokenized/airplane_train.pt
"""

import argparse
import os
import time

import torch
import torch.optim as optim
import yaml
from torch.utils.data import Dataset, DataLoader

from model.transformer import PointNSPTransformer


class TokenizedDataset(Dataset):
    """Dataset of pre-tokenized point clouds."""

    def __init__(self, path):
        data = torch.load(path, map_location="cpu")
        self.tokens = data["tokens"]  # dict: {k: (N, s_k)}
        self.coords = data["coords"]  # dict: {k: (N, s_k, 3)}
        self.scale_points = data["scale_points"]
        self.K = len(self.scale_points)
        self.N = self.tokens[0].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        tokens = [self.tokens[k][idx] for k in range(self.K)]
        coords = [self.coords[k][idx] for k in range(self.K)]
        return tokens, coords


def collate_fn(batch):
    """Collate list of (tokens, coords) into batched lists."""
    K = len(batch[0][0])
    tokens = [torch.stack([b[0][k] for b in batch]) for k in range(K)]
    coords = [torch.stack([b[1][k] for b in batch]) for k in range(K)]
    return tokens, coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenized_data", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Model
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
    print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    # Data
    dataset = TokenizedDataset(args.tokenized_data)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = optim.AdamW(
        transformer.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    num_epochs = train_cfg["num_epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Resume
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

    save_every = train_cfg.get("save_every", 50)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    for epoch in range(start_epoch, num_epochs):
        transformer.train()
        loss_accum = 0.0
        num_batches = 0
        t0 = time.time()

        for tokens, coords in dataloader:
            tokens = [t.to(device) for t in tokens]
            coords = [c.to(device) for c in coords]

            optimizer.zero_grad()
            loss = transformer.compute_loss(tokens, coords)
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
            torch.save(ckpt_data, os.path.join(args.output_dir, f"transformer_epoch{epoch + 1}.pt"))
            print(f"  -> Saved periodic checkpoint at epoch {epoch + 1}")

    print("Training complete.")


if __name__ == "__main__":
    main()

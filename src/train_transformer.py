import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from transformer_model import TransformerTrajectoryModel


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("outputs")


def build_parser():
    parser = argparse.ArgumentParser(description="Train Transformer baseline for trajectory prediction")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing processed .npy files")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, or cuda")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory for DataLoader")
    parser.add_argument("--input_dim", type=int, default=4, help="Input feature dimension")
    parser.add_argument("--d_model", type=int, default=64, help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=128, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--output_len", type=int, default=50, help="Number of future steps to predict")
    return parser


def resolve_device(device_name):
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return device_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_paths(data_dir):
    return data_dir / "X.npy", data_dir / "Y.npy"


def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


class TrajectoryDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], copy=False)).float()
        y = torch.from_numpy(np.array(self.Y[idx], copy=False)).float()
        return x, y


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        target = y[:, :, :2]

        pred = model(x)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target = y[:, :, :2]

            pred = model(x)
            loss = criterion(pred, target)

            total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    x_path, y_path = get_data_paths(args.data_dir)
    if not x_path.exists():
        raise FileNotFoundError(f"Missing X file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing Y file: {y_path}")

    checkpoint_dir = ensure_directory(args.output_dir / "checkpoints_transformer")

    dataset = TrajectoryDataset(x_path, y_path)

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Invalid split sizes: train_size={train_size}, val_size={val_size}")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
    )

    model = TransformerTrajectoryModel(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        output_len=args.output_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    print("=" * 60)
    print("Start Training Transformer")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = checkpoint_dir / "best_transformer_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best Transformer model to {save_path}")

    print("Transformer training finished.")


if __name__ == "__main__":
    main()

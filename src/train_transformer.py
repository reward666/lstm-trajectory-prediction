import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from transformer_model import TransformerTrajectoryModel


X_PATH = Path("data/processed/X.npy")
Y_PATH = Path("data/processed/Y.npy")

CHECKPOINT_DIR = Path("outputs/checkpoints_transformer")

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
VAL_RATIO = 0.2
DEVICE = "cpu"


class TrajectoryDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = TrajectoryDataset(X_PATH, Y_PATH)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TransformerTrajectoryModel(
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_len=50,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    print("=" * 60)
    print("Start Training Transformer")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CHECKPOINT_DIR / "best_transformer_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best Transformer model to {save_path}")

    print("Transformer training finished.")


if __name__ == "__main__":
    main()
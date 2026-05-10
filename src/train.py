'''import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from model import LSTMModel


X_PATH = Path("data/processed/X.npy")
Y_PATH = Path("data/processed/Y.npy")
CHECKPOINT_DIR = Path("outputs/checkpoints")

BATCH_SIZE = 128
EPOCHS = 10
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

    model = LSTMModel(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        output_len=50,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    print("=" * 60)
    print("Start Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")

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
            save_path = CHECKPOINT_DIR / "best_lstm_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from model import SocialLSTMWithAttention


X_PATH = Path("data/processed/X.npy")
Y_PATH = Path("data/processed/Y.npy")
X_SOCIAL_PATH = Path("data/processed/X_social.npy")
MASK_PATH = Path("data/processed/social_masks.npy")

CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_DIR = Path("outputs/logs")

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESUME = True


class TrajectoryDataset(Dataset):
    def __init__(self, x_path, y_path, x_social_path, mask_path):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)
        self.X_social = np.load(x_social_path)
        self.masks = np.load(mask_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        x_social = torch.tensor(self.X_social[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        return x, y, x_social, mask


def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for x, y, x_social, mask in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        x_social = x_social.to(DEVICE)
        mask = mask.to(DEVICE)

        pred = model(x, x_social, mask)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * x.size(0)

        pbar.set_postfix({
            "batch_loss": f"{loss.item():.6f}"
        })

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, epoch):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for x, y, x_social, mask in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x_social = x_social.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(x, x_social, mask)
            loss = criterion(pred, y)

            total_loss += loss.item() * x.size(0)

            pbar.set_postfix({
                "val_batch_loss": f"{loss.item():.6f}"
            })

    return total_loss / len(dataloader.dataset)


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def save_loss_curve(history, save_path):
    df = pd.DataFrame(history)
    df.to_csv(LOG_DIR / "loss_history.csv", index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    dataset = TrajectoryDataset(
        X_PATH,
        Y_PATH,
        X_SOCIAL_PATH,
        MASK_PATH
    )

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SocialLSTMWithAttention(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        output_len=50,
        num_heads=4,
        max_neighbors=5
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    start_epoch = 1
    best_val_loss = float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": []
    }

    last_ckpt_path = CHECKPOINT_DIR / "last_checkpoint.pt"
    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"

    if RESUME and last_ckpt_path.exists():
        checkpoint = torch.load(last_ckpt_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        history_path = LOG_DIR / "loss_history.csv"
        if history_path.exists():
            old_history = pd.read_csv(history_path)
            history = old_history.to_dict(orient="list")

        print(f"Resume training from epoch {start_epoch}")
        print(f"Best val loss so far: {best_val_loss:.6f}")

    print("=" * 60)
    print("Start Training: Social LSTM + Multi-Head Attention")
    print("=" * 60)
    print(f"Device:        {DEVICE}")
    print(f"Train samples: {train_size}")
    print(f"Val samples:   {val_size}")
    print(f"Total params:  {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            epoch
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        save_loss_curve(
            history,
            LOG_DIR / "loss_curve.png"
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            last_ckpt_path
        )

        print(
            f"\nEpoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        print("Saved last checkpoint.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_ckpt_path
            )

            print(f"Saved best model. Val Loss = {val_loss:.6f}")

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Loss history saved to: {LOG_DIR / 'loss_history.csv'}")
    print(f"Loss curve saved to:   {LOG_DIR / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
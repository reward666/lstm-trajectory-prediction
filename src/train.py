import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb

from model import SocialLSTMWithAttention


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("outputs")


def build_parser():
    parser = argparse.ArgumentParser(description="Train Social LSTM with Attention")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing processed .npy files")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoints, logs, and artifacts")
    parser.add_argument("--project", type=str, default="lstm_trajectory_prediction", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default=None, help="Optional Weights & Biases run name")
    parser.add_argument("--tags", type=str, nargs="*", default=[], help="Optional Weights & Biases tags")
    parser.add_argument("--notes", type=str, default="", help="Optional Weights & Biases notes")
    parser.add_argument("--wandb_mode", type=str, choices=("online", "offline", "disabled"), default="online", help="Weights & Biases mode")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory for DataLoader")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from last checkpoint if available")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of temporal attention heads")
    parser.add_argument("--input_dim", type=int, default=4, help="Input feature dimension")
    parser.add_argument("--output_len", type=int, default=50, help="Number of future steps to predict")
    parser.add_argument("--max_neighbors", type=int, default=5, help="Maximum number of neighboring vehicles")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
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
    return {
        "x": data_dir / "X.npy",
        "y": data_dir / "Y.npy",
        "x_social": data_dir / "X_social.npy",
        "mask": data_dir / "social_masks.npy",
    }


def ensure_directories(output_dir):
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, log_dir


def save_run_config(config, output_dir):
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, ensure_ascii=False, default=str)
    return config_path


class TrajectoryDataset(Dataset):
    def __init__(self, x_path, y_path, x_social_path, mask_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")
        self.X_social = np.load(x_social_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], copy=False)).float()
        y = torch.from_numpy(np.array(self.Y[idx], copy=False)).float()
        x_social = torch.from_numpy(np.array(self.X_social[idx], copy=False)).float()
        mask = torch.from_numpy(np.array(self.masks[idx], copy=False)).float()

        return x, y, x_social, mask

def multimodal_loss(
    pred_traj,
    log_prob,
    target_traj,
    tau=2.0,
    cls_weight=1.0,
    div_weight=0.1,
    ent_weight=0.002,
    div_margin=3.0,
):
    """
    Soft-WTA + Diversity + Entropy 多模态损失函数。
    pred_traj: [B, 6, 50, 2]     预测的 M 条未来轨迹
    log_prob:  [B, 6]            (Log_Softmax后)的 M 个概率
    target_traj: [B, 50, 4]      真实的未来轨迹 (X, Y, V, A) 
    """
    B, M, T, D_pred = pred_traj.shape
    
    # 只取 target_traj 的前两维 (X, Y) 计算坐标误差
    target_coord = target_traj[:, :, :2]
    
    target_traj_exp = target_coord.unsqueeze(1).expand(B, M, T, D_pred)
    
    # 算出每条预测轨迹和真实轨迹在未来 50 帧所有点上的 L2 误差
    # 对每条轨迹的所有时间步求和 (T, D) => [B, M]
    traj_mse = torch.sum((pred_traj - target_traj_exp) ** 2, dim=(2, 3))

    # Soft-WTA：让多个接近 GT 的 mode 都能分到梯度，避免 mode collapse
    responsibilities = torch.softmax(-traj_mse / tau, dim=1)
    reg_loss = (responsibilities.detach() * traj_mse).sum(dim=1).mean()

    # 分类项：让概率头拟合 soft responsibility 分布，而不是硬 winner
    cls_loss = -(responsibilities.detach() * log_prob).sum(dim=1).mean()

    # 多样性项：鼓励不同 mode 的终点彼此分离
    endpoints = pred_traj[:, :, -1, :]  # [B, M, 2]
    pairwise_dist = torch.cdist(endpoints, endpoints, p=2)
    upper_tri = torch.triu(torch.ones(M, M, device=pred_traj.device, dtype=torch.bool), diagonal=1)
    pairwise_dist = pairwise_dist[:, upper_tri]
    div_loss = torch.relu(div_margin - pairwise_dist).mean()

    # 熵项：防止概率过快单峰化
    prob = log_prob.exp()
    entropy = -(prob * log_prob).sum(dim=1).mean()
    entropy_loss = -entropy

    loss = reg_loss + cls_weight * cls_loss + div_weight * div_loss + ent_weight * entropy_loss

    metrics = {
        "reg_loss": reg_loss.detach(),
        "cls_loss": cls_loss.detach(),
        "div_loss": div_loss.detach(),
        "entropy": entropy.detach(),
        "mode_spread": pairwise_dist.mean().detach(),
    }
    return loss, metrics


def train_one_epoch(model, dataloader, optimizer, epoch, device, grad_clip):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for x, y, x_social, mask in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_social = x_social.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred_traj, pred_prob = model(x, x_social, mask)
        loss, metrics = multimodal_loss(pred_traj, pred_prob, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item() * x.size(0)

        pbar.set_postfix({
            "batch_loss": f"{loss.item():.6f}",
            "spread": f"{metrics['mode_spread'].item():.4f}"
        })

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, epoch, device):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for x, y, x_social, mask in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_social = x_social.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            pred_traj, pred_prob = model(x, x_social, mask)
            loss, metrics = multimodal_loss(pred_traj, pred_prob, y)

            total_loss += loss.item() * x.size(0)

            pbar.set_postfix({
                "val_batch_loss": f"{loss.item():.6f}",
                "spread": f"{metrics['mode_spread'].item():.4f}"
            })

    return total_loss / len(dataloader.dataset)


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
    }, path)


def save_loss_curve(history, save_path):
    df = pd.DataFrame(history)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path.parent / "loss_history.csv", index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("WTA Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    paths = get_data_paths(args.data_dir)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} file: {path}")

    checkpoint_dir, log_dir = ensure_directories(args.output_dir)
    config = vars(args).copy()
    config["device"] = device
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["log_dir"] = str(log_dir)
    save_run_config(config, args.output_dir)

    dataset = TrajectoryDataset(paths["x"], paths["y"], paths["x_social"], paths["mask"])

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Invalid split sizes: train_size={train_size}, val_size={val_size}")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = SocialLSTMWithAttention(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_len=args.output_len,
        num_heads=args.num_heads,
        max_neighbors=args.max_neighbors,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
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

    last_ckpt_path = checkpoint_dir / "last_checkpoint.pt"
    best_ckpt_path = checkpoint_dir / "best_model.pt"

    wandb_run = wandb.init(
        project=args.project,
        name=args.run_name,
        config=config,
        dir=str(args.output_dir),
        mode=args.wandb_mode,
        tags=args.tags or None,
        notes=args.notes or None,
        reinit=True,
    )

    if args.resume and last_ckpt_path.exists():
        checkpoint = torch.load(last_ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        history_path = log_dir / "loss_history.csv"
        if history_path.exists():
            old_history = pd.read_csv(history_path)
            history = old_history.to_dict(orient="list")

        print(f"Resume training from epoch {start_epoch}")
        print(f"Best val loss so far: {best_val_loss:.6f}")

    print("=" * 60)
    print("Start Training: Social LSTM + Multi-Head Attention")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Train samples: {train_size}")
    print(f"Val samples:   {val_size}")
    print(f"Total params:  {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            device,
            args.grad_clip,
        )

        val_loss = evaluate(
            model,
            val_loader,
            epoch,
            device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "epoch": epoch
        })

        save_loss_curve(
            history,
            log_dir / "loss_curve.png"
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            last_ckpt_path,
            config,
        )

        print(
            f"\nEpoch [{epoch:02d}/{args.epochs}] "
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
                best_ckpt_path,
                config,
            )

            print(f"Saved best model. Val Loss = {val_loss:.6f}")

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Loss history saved to: {log_dir / 'loss_history.csv'}")
    print(f"Loss curve saved to:   {log_dir / 'loss_curve.png'}")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

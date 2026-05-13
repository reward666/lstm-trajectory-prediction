import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from model import SocialLSTMWithAttention
from transformer_model import TransformerTrajectoryModel


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_TRANSFORMER_CKPT = DEFAULT_OUTPUT_DIR / "checkpoints_transformer" / "best_transformer_model.pt"
DEFAULT_SOCIAL_CKPT = DEFAULT_OUTPUT_DIR / "checkpoints" / "best_model.pt"


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


def build_parser():
    parser = argparse.ArgumentParser(description="Compare Transformer baseline and Social LSTM on the same validation split")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing processed .npy files")
    parser.add_argument("--transformer_ckpt", type=Path, default=DEFAULT_TRANSFORMER_CKPT, help="Path to the Transformer checkpoint")
    parser.add_argument("--social_ckpt", type=Path, default=DEFAULT_SOCIAL_CKPT, help="Path to the Social LSTM checkpoint")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for comparison artifacts")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory for DataLoader")
    parser.add_argument("--transformer_input_dim", type=int, default=4, help="Transformer input dimension")
    parser.add_argument("--transformer_d_model", type=int, default=64, help="Transformer hidden dimension")
    parser.add_argument("--transformer_nhead", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--transformer_num_layers", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--transformer_dim_feedforward", type=int, default=128, help="Transformer feedforward dimension")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("--transformer_output_len", type=int, default=50, help="Transformer prediction horizon")
    parser.add_argument("--social_input_dim", type=int, default=4, help="Social LSTM input dimension")
    parser.add_argument("--social_hidden_dim", type=int, default=64, help="Social LSTM hidden dimension")
    parser.add_argument("--social_num_layers", type=int, default=2, help="Social LSTM layers")
    parser.add_argument("--social_num_heads", type=int, default=4, help="Social attention heads")
    parser.add_argument("--social_output_len", type=int, default=50, help="Social LSTM prediction horizon")
    parser.add_argument("--social_max_neighbors", type=int, default=5, help="Maximum neighboring vehicles")
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


def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {})
    return checkpoint, {}


def compute_ade_fde(pred, target):
    diff = pred - target
    dist = torch.norm(diff, dim=-1)
    ade = dist.mean(dim=1)
    fde = dist[:, -1]
    return ade, fde


def evaluate_transformer(model, dataloader, device):
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y, _, _ in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target = y[:, :, :2]

            pred = model(x)
            ade, fde = compute_ade_fde(pred, target)

            batch_size = x.size(0)
            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            total_samples += batch_size

    return {
        "ADE": total_ade / total_samples,
        "FDE": total_fde / total_samples,
    }


def evaluate_social(model, dataloader, device):
    model.eval()
    total_selected_ade = 0.0
    total_selected_fde = 0.0
    total_oracle_ade = 0.0
    total_oracle_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y, x_social, mask in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_social = x_social.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = y[:, :, :2]

            pred_traj, pred_prob = model(x, x_social, mask)
            best_mode = pred_prob.argmax(dim=-1)
            batch_indices = torch.arange(pred_traj.size(0), device=device)
            selected_pred = pred_traj[batch_indices, best_mode]

            selected_ade, selected_fde = compute_ade_fde(selected_pred, target)

            all_mode_ade = torch.norm(pred_traj - target.unsqueeze(1), dim=-1).mean(dim=-1)
            all_mode_fde = torch.norm(pred_traj[:, :, -1, :] - target[:, -1, :].unsqueeze(1), dim=-1)
            oracle_idx = all_mode_ade.argmin(dim=1)
            oracle_pred = pred_traj[batch_indices, oracle_idx]
            oracle_ade, oracle_fde = compute_ade_fde(oracle_pred, target)

            batch_size = x.size(0)
            total_selected_ade += selected_ade.sum().item()
            total_selected_fde += selected_fde.sum().item()
            total_oracle_ade += oracle_ade.sum().item()
            total_oracle_fde += oracle_fde.sum().item()
            total_samples += batch_size

    return {
        "ADE": total_selected_ade / total_samples,
        "FDE": total_selected_fde / total_samples,
        "oracle_ADE": total_oracle_ade / total_samples,
        "oracle_FDE": total_oracle_fde / total_samples,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    paths = get_data_paths(args.data_dir)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} file: {path}")

    if not args.transformer_ckpt.exists():
        raise FileNotFoundError(f"Missing Transformer checkpoint: {args.transformer_ckpt}")
    if not args.social_ckpt.exists():
        raise FileNotFoundError(f"Missing Social LSTM checkpoint: {args.social_ckpt}")

    output_dir = ensure_directory(args.output_dir / "comparison")

    dataset = TrajectoryDataset(paths["x"], paths["y"], paths["x_social"], paths["mask"])
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Invalid split sizes: train_size={train_size}, val_size={val_size}")

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    persistent_workers = args.num_workers > 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
    )

    transformer = TransformerTrajectoryModel(
        input_dim=args.transformer_input_dim,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_layers=args.transformer_num_layers,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout,
        output_len=args.transformer_output_len,
    ).to(device)
    transformer_state_dict, transformer_config = load_checkpoint_state(args.transformer_ckpt, device)
    transformer.load_state_dict(transformer_state_dict)

    social = SocialLSTMWithAttention(
        input_dim=args.social_input_dim,
        hidden_dim=args.social_hidden_dim,
        num_layers=args.social_num_layers,
        output_len=args.social_output_len,
        num_heads=args.social_num_heads,
        max_neighbors=args.social_max_neighbors,
    ).to(device)
    social_state_dict, social_config = load_checkpoint_state(args.social_ckpt, device)
    social.load_state_dict(social_state_dict)

    transformer_metrics = evaluate_transformer(transformer, val_loader, device)
    social_metrics = evaluate_social(social, val_loader, device)

    report = {
        "device": device,
        "seed": args.seed,
        "val_samples": val_size,
        "transformer": {
            "checkpoint": str(args.transformer_ckpt),
            "params": sum(p.numel() for p in transformer.parameters()),
            "config": transformer_config,
            **transformer_metrics,
        },
        "social_lstm": {
            "checkpoint": str(args.social_ckpt),
            "params": sum(p.numel() for p in social.parameters()),
            "config": social_config,
            **social_metrics,
        },
    }

    report_path = output_dir / "comparison_report.json"
    csv_path = output_dir / "comparison_report.csv"

    with report_path.open("w", encoding="utf-8") as file_obj:
        import json
        json.dump(report, file_obj, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("model,params,ADE,FDE,oracle_ADE,oracle_FDE\n")
        file_obj.write(
            f"transformer,{report['transformer']['params']},{report['transformer']['ADE']:.6f},{report['transformer']['FDE']:.6f},,\n"
        )
        file_obj.write(
            f"social_lstm,{report['social_lstm']['params']},{report['social_lstm']['ADE']:.6f},{report['social_lstm']['FDE']:.6f},{report['social_lstm']['oracle_ADE']:.6f},{report['social_lstm']['oracle_FDE']:.6f}\n"
        )

    print("=" * 72)
    print("Baseline vs Enhanced Comparison")
    print("=" * 72)
    print(f"Validation samples: {val_size}")
    print(f"Transformer params: {report['transformer']['params']:,}")
    print(f"Transformer ADE/FDE: {report['transformer']['ADE']:.6f} / {report['transformer']['FDE']:.6f}")
    print(f"Social LSTM params: {report['social_lstm']['params']:,}")
    print(f"Social LSTM ADE/FDE: {report['social_lstm']['ADE']:.6f} / {report['social_lstm']['FDE']:.6f}")
    print(f"Social LSTM oracle ADE/FDE: {report['social_lstm']['oracle_ADE']:.6f} / {report['social_lstm']['oracle_FDE']:.6f}")
    print(f"Saved JSON report to: {report_path}")
    print(f"Saved CSV report to:  {csv_path}")


if __name__ == "__main__":
    main()

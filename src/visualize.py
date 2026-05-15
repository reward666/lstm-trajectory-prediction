import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import SocialLSTMWithAttention
from transformer_model import TransformerTrajectoryModel


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_SOCIAL_CKPT = DEFAULT_OUTPUT_DIR / "checkpoints" / "best_model.pt"
DEFAULT_TRANSFORMER_CKPT = DEFAULT_OUTPUT_DIR / "checkpoints_transformer" / "best_transformer_model.pt"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize Social LSTM and Transformer predictions.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing processed .npy files")
    parser.add_argument("--social_ckpt", type=Path, default=DEFAULT_SOCIAL_CKPT, help="Path to the Social LSTM checkpoint")
    parser.add_argument("--transformer_ckpt", type=Path, default=DEFAULT_TRANSFORMER_CKPT, help="Path to the Transformer checkpoint")
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIGURE_DIR, help="Directory for generated figures")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu or cuda")
    parser.add_argument("--indices", type=int, nargs="*", default=[0, 1000, 5000, 10000, 20000], help="Sample indices to plot")
    return parser


def load_checkpoint_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {})
    return checkpoint, {}


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_social_model(checkpoint_path, input_dim, max_neighbors, device):
    state_dict, config = load_checkpoint_state(checkpoint_path, device)
    model = SocialLSTMWithAttention(
        input_dim=config.get("input_dim", input_dim),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        output_len=config.get("output_len", 50),
        num_heads=config.get("num_heads", 4),
        max_neighbors=config.get("max_neighbors", max_neighbors),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_transformer_model(checkpoint_path, input_dim, device):
    state_dict, config = load_checkpoint_state(checkpoint_path, device)
    model = TransformerTrajectoryModel(
        input_dim=config.get("input_dim", input_dim),
        d_model=config.get("d_model", 64),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
        num_encoder_layers=config.get("num_encoder_layers"),
        num_decoder_layers=config.get("num_decoder_layers"),
        dim_feedforward=config.get("dim_feedforward", 128),
        dropout=config.get("dropout", 0.1),
        output_len=config.get("output_len", 50),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_one_sample(index, X, Y, X_social, masks, social_model, transformer_model, figure_dir, device):
    hist = X[index]
    future = Y[index]
    social = X_social[index]
    mask = masks[index]

    x_tensor = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(device)
    social_tensor = torch.tensor(social, dtype=torch.float32).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        social_trajs, social_log_probs = social_model(x_tensor, social_tensor, mask_tensor)
        selected_mode = social_log_probs.argmax(dim=-1).item()
        social_pred = social_trajs[0, selected_mode].cpu().numpy()
        transformer_pred = transformer_model(x_tensor).squeeze(0).cpu().numpy()

    plt.figure(figsize=(8, 6))

    plt.plot(hist[:, 0], hist[:, 1], marker="o", label="History")
    plt.plot(future[:, 0], future[:, 1], marker="o", label="Ground Truth")
    plt.plot(social_pred[:, 0], social_pred[:, 1], marker="x", label=f"Social LSTM Prediction (mode {selected_mode})")
    plt.plot(transformer_pred[:, 0], transformer_pred[:, 1], marker="x", label="Transformer Prediction")

    plt.xlabel("Local X relative position (m)")
    plt.ylabel("Local Y relative position (m)")
    plt.title(f"Trajectory Prediction Comparison - Sample {index}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    save_path = figure_dir / f"compare_sample_{index}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {save_path}")


def main():
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    x_path = args.data_dir / "X.npy"
    y_path = args.data_dir / "Y.npy"
    x_social_path = args.data_dir / "X_social.npy"
    mask_path = args.data_dir / "social_masks.npy"

    require_file(x_path, "X file")
    require_file(y_path, "Y file")
    require_file(x_social_path, "X_social file")
    require_file(mask_path, "social mask file")
    require_file(args.social_ckpt, "Social LSTM checkpoint")
    require_file(args.transformer_ckpt, "Transformer checkpoint")

    args.figure_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    X_social = np.load(x_social_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")

    print("Loading models...")
    social_model = load_social_model(args.social_ckpt, input_dim=X.shape[-1], max_neighbors=X_social.shape[1], device=args.device)
    transformer_model = load_transformer_model(args.transformer_ckpt, input_dim=X.shape[-1], device=args.device)

    valid_indices = [idx for idx in args.indices if 0 <= idx < len(X)]
    skipped_indices = sorted(set(args.indices) - set(valid_indices))
    if skipped_indices:
        print(f"Skipping out-of-range sample indices: {skipped_indices}")
    if not valid_indices:
        raise ValueError(f"No valid sample indices were provided for dataset length {len(X)}.")

    for idx in valid_indices:
        plot_one_sample(idx, X, Y, X_social, masks, social_model, transformer_model, args.figure_dir, args.device)

    print("Visualization finished.")


if __name__ == "__main__":
    main()

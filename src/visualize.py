import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from model import LSTMModel
from transformer_model import TransformerTrajectoryModel


X_PATH = Path("data/processed/X.npy")
Y_PATH = Path("data/processed/Y.npy")

LSTM_CKPT = Path("outputs/checkpoints/best_lstm_model.pt")
TRANSFORMER_CKPT = Path("outputs/checkpoints_transformer/best_transformer_model.pt")

FIGURE_DIR = Path("outputs/figures")
DEVICE = "cpu"


def load_lstm_model():
    model = LSTMModel(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        output_len=50,
    ).to(DEVICE)

    model.load_state_dict(torch.load(LSTM_CKPT, map_location=DEVICE))
    model.eval()
    return model


def load_transformer_model():
    model = TransformerTrajectoryModel(
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_len=50,
    ).to(DEVICE)

    model.load_state_dict(torch.load(TRANSFORMER_CKPT, map_location=DEVICE))
    model.eval()
    return model


def plot_one_sample(index, X, Y, lstm_model, transformer_model):
    hist = X[index]      # [30, 2]
    future = Y[index]    # [50, 2]

    x_tensor = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        lstm_pred = lstm_model(x_tensor).squeeze(0).cpu().numpy()
        transformer_pred = transformer_model(x_tensor).squeeze(0).cpu().numpy()

    plt.figure(figsize=(8, 6))

    plt.plot(hist[:, 0], hist[:, 1], marker="o", label="History")
    plt.plot(future[:, 0], future[:, 1], marker="o", label="Ground Truth")
    plt.plot(lstm_pred[:, 0], lstm_pred[:, 1], marker="x", label="LSTM Prediction")
    plt.plot(transformer_pred[:, 0], transformer_pred[:, 1], marker="x", label="Transformer Prediction")

    plt.xlabel("Local X relative position (m)")
    plt.ylabel("Local Y relative position (m)")
    plt.title(f"Trajectory Prediction Comparison - Sample {index}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    save_path = FIGURE_DIR / f"compare_sample_{index}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {save_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    print("Loading models...")
    lstm_model = load_lstm_model()
    transformer_model = load_transformer_model()

    # 先随机选几个样本画图
    sample_indices = [0, 1000, 5000, 10000, 20000]

    for idx in sample_indices:
        plot_one_sample(idx, X, Y, lstm_model, transformer_model)

    print("Visualization finished.")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_PROCESSED_DATA_PATH, FUTURE_LEN, HISTORY_LEN, PROCESSED_DATA_DIR, STEP


def build_trajectory_samples(df: pd.DataFrame, history_len=HISTORY_LEN, future_len=FUTURE_LEN, step=STEP):
    """
    把原始逐帧数据变成 LSTM 训练样本。

    每个样本：
        X: 过去 history_len 帧的位置 [history_len, 2]
        Y: 未来 future_len 帧的位置 [future_len, 2]
    """

    X_list = []
    Y_list = []

    grouped = df.groupby("Vehicle_ID")

    for _, vehicle_data in grouped:
        vehicle_data = vehicle_data.sort_values("Frame_ID")
        traj = vehicle_data[["Local_X", "Local_Y"]].values

        total_len = len(traj)
        needed_len = history_len + future_len

        if total_len < needed_len:
            continue

        for start in range(0, total_len - needed_len + 1, step):
            hist = traj[start: start + history_len]
            fut = traj[start + history_len: start + history_len + future_len]

            origin = hist[-1].copy()

            hist_rel = hist - origin
            fut_rel = fut - origin

            X_list.append(hist_rel)
            Y_list.append(fut_rel)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    return X, Y


def build_parser():
    parser = argparse.ArgumentParser(description="Build simple trajectory samples from processed data.")
    parser.add_argument("--processed_path", type=Path, default=DEFAULT_PROCESSED_DATA_PATH, help="Input processed .pkl path")
    parser.add_argument("--output_dir", type=Path, default=PROCESSED_DATA_DIR, help="Directory for X.npy and Y.npy")
    parser.add_argument("--history_len", type=int, default=HISTORY_LEN, help="Historical frame count")
    parser.add_argument("--future_len", type=int, default=FUTURE_LEN, help="Future frame count")
    parser.add_argument("--step", type=int, default=STEP, help="Sliding window step")
    return parser


def main():
    args = build_parser().parse_args()

    print(f"Loading processed data from: {args.processed_path}")
    if not args.processed_path.exists():
        raise SystemExit(
            f"处理后文件不存在: {args.processed_path}\n"
            "请先运行 `python src/preprocess.py --raw_path <RAW_TXT>`。"
        )

    df = pd.read_pickle(args.processed_path)

    print("Building trajectory samples...")
    X, Y = build_trajectory_samples(df, args.history_len, args.future_len, args.step)

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.output_dir / "X.npy", X)
    np.save(args.output_dir / "Y.npy", Y)

    print("\nSaved:")
    print(args.output_dir / "X.npy")
    print(args.output_dir / "Y.npy")


if __name__ == "__main__":
    main()

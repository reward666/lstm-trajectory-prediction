import pandas as pd
import numpy as np
from pathlib import Path


PROCESSED_DATA_PATH = Path("data/processed/ngsim_us101_0750_0805_processed.pkl")
OUTPUT_DIR = Path("data/processed")

HISTORY_LEN = 30   # 过去 3 秒，NGSIM 是 10Hz
FUTURE_LEN = 50    # 未来 5 秒
STEP = 5           # 滑动窗口步长，先设为 5，减少样本量


def build_trajectory_samples(df: pd.DataFrame):
    """
    把原始逐帧数据变成 LSTM 训练样本。

    每个样本：
        X: 过去 HISTORY_LEN 帧的位置 [30, 2]
        Y: 未来 FUTURE_LEN 帧的位置 [50, 2]
    """

    X_list = []
    Y_list = []

    # 按车辆 ID 分组，每辆车单独构造轨迹
    grouped = df.groupby("Vehicle_ID")

    for vehicle_id, vehicle_data in grouped:
        # 保证每辆车内部按时间排序
        vehicle_data = vehicle_data.sort_values("Frame_ID")

        # 只取位置坐标
        traj = vehicle_data[["Local_X", "Local_Y"]].values

        # 轨迹长度不够就跳过
        total_len = len(traj)
        needed_len = HISTORY_LEN + FUTURE_LEN

        if total_len < needed_len:
            continue

        # 滑动窗口构造样本
        for start in range(0, total_len - needed_len + 1, STEP):
            hist = traj[start : start + HISTORY_LEN]
            fut = traj[start + HISTORY_LEN : start + HISTORY_LEN + FUTURE_LEN]

            # 以历史轨迹最后一个点作为原点，做相对坐标
            origin = hist[-1].copy()

            hist_rel = hist - origin
            fut_rel = fut - origin

            X_list.append(hist_rel)
            Y_list.append(fut_rel)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    return X, Y


def main():
    print(f"Loading processed data from: {PROCESSED_DATA_PATH}")

    df = pd.read_pickle(PROCESSED_DATA_PATH)

    print("Building trajectory samples...")
    X, Y = build_trajectory_samples(df)

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "Y.npy", Y)

    print("\nSaved:")
    print(OUTPUT_DIR / "X.npy")
    print(OUTPUT_DIR / "Y.npy")


if __name__ == "__main__":
    main()
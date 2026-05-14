import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_PROCESSED_DATA_PATH,
    FUTURE_LEN,
    HISTORY_LEN,
    MAX_NEIGHBORS,
    NEIGHBOR_RADIUS,
    PROCESSED_DATA_DIR,
    STEP,
)

# 全局变量，子进程共享
frame_index = {}
df_global = None


def build_frame_index(df):
    index = {}
    for row in df.itertuples():
        fid = row.Frame_ID
        if fid not in index:
            index[fid] = {}
        # 提取更多特征: (Local_X, Local_Y, v_Vel, v_Acc)
        index[fid][row.Vehicle_ID] = (row.Local_X, row.Local_Y, row.v_Vel, row.v_Acc)
    return index


def get_neighbor_ids(anchor_frame_id, ego_id, origin):
    frame_group = frame_index.get(anchor_frame_id, {})
    ego_x, ego_y = origin[0], origin[1]
    neighbors = []
    for vid, features in frame_group.items():
        if vid == ego_id:
            continue
        x, y = features[0], features[1]
        dist = np.sqrt((x - ego_x) ** 2 + (y - ego_y) ** 2)
        if dist < NEIGHBOR_RADIUS:
            neighbors.append((dist, vid))
    neighbors.sort(key=lambda t: t[0])
    return [vid for _, vid in neighbors[:MAX_NEIGHBORS]]


def process_vehicle(args):
    """每辆车单独处理，在子进程中运行"""
    vehicle_id, vehicle_data_records = args

    vehicle_data = pd.DataFrame(vehicle_data_records).sort_values("Frame_ID").reset_index(drop=True)
    
    # 获取特征: X, Y, 速度, 加速度
    traj = vehicle_data[["Local_X", "Local_Y", "v_Vel", "v_Acc"]].values
    frame_ids = vehicle_data["Frame_ID"].values

    total_len = len(traj)
    needed_len = HISTORY_LEN + FUTURE_LEN
    if total_len < needed_len:
        return None

    X_list, Y_list, X_social_list, mask_list = [], [], [], []

    # 预建邻居数据索引 {vehicle_id: {frame_id: (x, y)}}
    # 只在当前车辆的帧范围内查找，避免重复查全表
    relevant_frame_ids = set(frame_ids)
    neighbor_data_cache = {}  # {vid: {fid: (x, y)}}

    for start in range(0, total_len - needed_len + 1, STEP):
        hist = traj[start: start + HISTORY_LEN]
        fut = traj[start + HISTORY_LEN: start + HISTORY_LEN + FUTURE_LEN]

        # 原点是最后时刻的位置 (X, Y)
        origin = hist[-1, :2].copy()
        
        hist_rel = hist.copy()
        fut_rel = fut.copy()
        
        # 只在 X, Y 维度做相对坐标变换，速度和加速度(维度2和3)保持绝对值或原本数值
        hist_rel[:, :2] = hist[:, :2] - origin
        fut_rel[:, :2] = fut[:, :2] - origin

        anchor_frame_id = frame_ids[start + HISTORY_LEN - 1]
        neighbor_ids = get_neighbor_ids(anchor_frame_id, vehicle_id, origin)

        social_trajs = []
        for nid in neighbor_ids:
            # … (缓存机制略过)
            if nid not in neighbor_data_cache:
                n_frame_group = {}
                for fid in relevant_frame_ids:
                    if fid in frame_index and nid in frame_index[fid]:
                        n_frame_group[fid] = frame_index[fid][nid]
                neighbor_data_cache[nid] = n_frame_group

            n_frame_map = neighbor_data_cache[nid]
            n_traj = []
            for fid in frame_ids[start: start + HISTORY_LEN]:
                if fid in n_frame_map:
                    n_traj.append(n_frame_map[fid])
                else:
                    # 如果缺失邻居帧，补一个和原点位置相同且速度、加速度为 0 的静止记录
                    n_traj.append((origin[0], origin[1], 0.0, 0.0))
            
            n_traj = np.array(n_traj, dtype=np.float32)
            n_traj[:, :2] = n_traj[:, :2] - origin
            social_trajs.append(n_traj)

        mask = [1] * len(social_trajs) + [0] * (MAX_NEIGHBORS - len(social_trajs))
        while len(social_trajs) < MAX_NEIGHBORS:
            # Pading 特征变成了 4 维 (X,Y,V,A)
            social_trajs.append(np.zeros((HISTORY_LEN, 4), dtype=np.float32))

        social_trajs = np.stack(social_trajs, axis=0)

        X_list.append(hist_rel)
        Y_list.append(fut_rel)
        X_social_list.append(social_trajs)
        mask_list.append(mask)

    if not X_list:
        return None

    return (
        np.array(X_list, dtype=np.float32),
        np.array(Y_list, dtype=np.float32),
        np.array(X_social_list, dtype=np.float32),
        np.array(mask_list, dtype=np.float32),
    )


def init_worker(fi):
    """子进程初始化，注入全局 frame_index"""
    global frame_index
    frame_index = fi


def build_parser():
    parser = argparse.ArgumentParser(description="Build social trajectory samples from processed data.")
    parser.add_argument(
        "--processed_path",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_PATH,
        help="Input processed .pkl path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for .npy sample files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Multiprocessing worker count",
    )
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

    print("Building frame index...")
    fi = build_frame_index(df)

    # 按车辆分组，转成 records 列表传给子进程
    print("Preparing vehicle groups...")
    vehicle_groups = [
        (vid, group.to_dict("records"))
        for vid, group in df.groupby("Vehicle_ID")
    ]

    num_workers = max(1, args.num_workers)
    print(f"Using {num_workers} workers (total CPUs: {cpu_count()})")
    print(f"Total vehicles: {len(vehicle_groups)}")

    print("Building trajectory samples (multiprocessing)...")
    with Pool(processes=num_workers, initializer=init_worker, initargs=(fi,)) as pool:
        results = pool.map(process_vehicle, vehicle_groups)

    # 过滤空结果并合并
    results = [r for r in results if r is not None]
    print(f"Valid vehicles: {len(results)}")
    if not results:
        raise SystemExit("没有生成有效样本，请检查数据长度或窗口参数。")

    X = np.concatenate([r[0] for r in results], axis=0)
    Y = np.concatenate([r[1] for r in results], axis=0)
    X_social = np.concatenate([r[2] for r in results], axis=0)
    masks = np.concatenate([r[3] for r in results], axis=0)

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"X shape:        {X.shape}")
    print(f"Y shape:        {Y.shape}")
    print(f"X_social shape: {X_social.shape}")
    print(f"masks shape:    {masks.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X.npy", X)
    np.save(args.output_dir / "Y.npy", Y)
    np.save(args.output_dir / "X_social.npy", X_social)
    np.save(args.output_dir / "social_masks.npy", masks)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

PROCESSED_DATA_PATH = Path("data/processed/ngsim_us101_0750_0805_processed.pkl")
OUTPUT_DIR = Path("data/processed")

HISTORY_LEN = 30
FUTURE_LEN = 50
STEP = 5
MAX_NEIGHBORS = 5
NEIGHBOR_RADIUS = 30.0

# 全局变量，子进程共享
frame_index = {}
df_global = None


def build_frame_index(df):
    index = {}
    for row in df.itertuples():
        fid = row.Frame_ID
        if fid not in index:
            index[fid] = {}
        index[fid][row.Vehicle_ID] = (row.Local_X, row.Local_Y)
    return index


def get_neighbor_ids(anchor_frame_id, ego_id, origin):
    frame_group = frame_index.get(anchor_frame_id, {})
    ego_x, ego_y = origin
    neighbors = []
    for vid, (x, y) in frame_group.items():
        if vid == ego_id:
            continue
        dist = np.sqrt((x - ego_x) ** 2 + (y - ego_y) ** 2)
        if dist < NEIGHBOR_RADIUS:
            neighbors.append((dist, vid))
    neighbors.sort(key=lambda t: t[0])
    return [vid for _, vid in neighbors[:MAX_NEIGHBORS]]


def process_vehicle(args):
    """每辆车单独处理，在子进程中运行"""
    vehicle_id, vehicle_data_records = args

    vehicle_data = pd.DataFrame(vehicle_data_records).sort_values("Frame_ID").reset_index(drop=True)
    traj = vehicle_data[["Local_X", "Local_Y"]].values
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

        origin = hist[-1].copy()
        hist_rel = hist - origin
        fut_rel = fut - origin

        anchor_frame_id = frame_ids[start + HISTORY_LEN - 1]
        neighbor_ids = get_neighbor_ids(anchor_frame_id, vehicle_id, origin)

        social_trajs = []
        for nid in neighbor_ids:
            # 懒加载邻居缓存
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
                    n_traj.append(tuple(origin))
            n_traj = np.array(n_traj, dtype=np.float32) - origin
            social_trajs.append(n_traj)

        mask = [1] * len(social_trajs) + [0] * (MAX_NEIGHBORS - len(social_trajs))
        while len(social_trajs) < MAX_NEIGHBORS:
            social_trajs.append(np.zeros((HISTORY_LEN, 2), dtype=np.float32))

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


def main():
    print(f"Loading processed data from: {PROCESSED_DATA_PATH}")
    df = pd.read_pickle(PROCESSED_DATA_PATH)

    print("Building frame index...")
    fi = build_frame_index(df)

    # 按车辆分组，转成 records 列表传给子进程
    print("Preparing vehicle groups...")
    vehicle_groups = [
        (vid, group.to_dict("records"))
        for vid, group in df.groupby("Vehicle_ID")
    ]

    num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} workers (total CPUs: {cpu_count()})")
    print(f"Total vehicles: {len(vehicle_groups)}")

    print("Building trajectory samples (multiprocessing)...")
    with Pool(processes=num_workers, initializer=init_worker, initargs=(fi,)) as pool:
        results = pool.map(process_vehicle, vehicle_groups)

    # 过滤空结果并合并
    results = [r for r in results if r is not None]
    print(f"Valid vehicles: {len(results)}")

    X        = np.concatenate([r[0] for r in results], axis=0)
    Y        = np.concatenate([r[1] for r in results], axis=0)
    X_social = np.concatenate([r[2] for r in results], axis=0)
    masks    = np.concatenate([r[3] for r in results], axis=0)

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"X shape:        {X.shape}")
    print(f"Y shape:        {Y.shape}")
    print(f"X_social shape: {X_social.shape}")
    print(f"masks shape:    {masks.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "Y.npy", Y)
    np.save(OUTPUT_DIR / "X_social.npy", X_social)
    np.save(OUTPUT_DIR / "social_masks.npy", masks)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
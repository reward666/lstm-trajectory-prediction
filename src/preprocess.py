import argparse
from pathlib import Path

import pandas as pd

from config import DEFAULT_PROCESSED_DATA_PATH, DEFAULT_RAW_DATA_PATH, FEET_TO_METER

NGSIM_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "Total_Frames",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Global_X",
    "Global_Y",
    "v_Length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
]

COLUMN_ALIASES = {
    "vehicle_id": "Vehicle_ID",
    "frame_id": "Frame_ID",
    "total_frames": "Total_Frames",
    "global_time": "Global_Time",
    "local_x": "Local_X",
    "local_y": "Local_Y",
    "global_x": "Global_X",
    "global_y": "Global_Y",
    "v_length": "v_Length",
    "v_width": "v_Width",
    "v_class": "v_Class",
    "v_vel": "v_Vel",
    "v_acc": "v_Acc",
    "lane_id": "Lane_ID",
    "preceding": "Preceding",
    "following": "Following",
    "space_headway": "Space_Headway",
    "time_headway": "Time_Headway",
}


def has_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        first_line = file_obj.readline().strip().lower()
    return any(alias in first_line for alias in COLUMN_ALIASES)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized]
    return df.rename(columns=rename_map)


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if has_header(path):
        df = pd.read_csv(path)
        return normalize_column_names(df)

    df = pd.read_csv(
        path,
        sep=r"\s+",     # 空格分隔（关键）
        header=None,    # 没有表头（关键）
        names=NGSIM_COLUMNS,  # 手动指定列名（关键）
        engine="python",
    )

    return df


def clean_ngsim_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "Vehicle_ID",
        "Frame_ID",
        "Total_Frames",
        "Global_Time",
        "Local_X",
        "Local_Y",
        "v_Vel",
        "v_Acc",
        "v_Length",
        "v_Width",
        "v_Class",
        "Lane_ID",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df[required_cols].copy()

    df = df.dropna()

    df["Local_X"] = df["Local_X"] * FEET_TO_METER
    df["Local_Y"] = df["Local_Y"] * FEET_TO_METER
    df["v_Vel"] = df["v_Vel"] * FEET_TO_METER
    df["v_Acc"] = df["v_Acc"] * FEET_TO_METER
    df["v_Length"] = df["v_Length"] * FEET_TO_METER
    df["v_Width"] = df["v_Width"] * FEET_TO_METER

    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Processed NGSIM Data Summary")
    print("=" * 60)

    print(f"Data shape: {df.shape}")
    print(f"Number of vehicles: {df['Vehicle_ID'].nunique()}")
    print(f"Frame range: {df['Frame_ID'].min()} -> {df['Frame_ID'].max()}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nBasic statistics:")
    print(df[["Local_X", "Local_Y", "v_Vel", "v_Acc"]].describe())


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)
    print(f"\nSaved processed data to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Preprocess raw NGSIM trajectory text/CSV data.")
    parser.add_argument(
        "--raw_path",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to the raw NGSIM trajectory .txt or .csv file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_PATH,
        help="Path for the processed .pkl file",
    )
    return parser


def main():
    args = build_parser().parse_args()

    print(f"Loading raw data from: {args.raw_path}")

    try:
        df_raw = load_raw_data(args.raw_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n"
            "请先运行 `python src/download_data.py --url <DATA_URL>` 下载数据，"
            "或用 `--raw_path` 指向服务器上的原始轨迹文件。"
        ) from exc

    print(f"Raw data shape: {df_raw.shape}")

    df_processed = clean_ngsim_data(df_raw)

    print_summary(df_processed)

    save_processed_data(df_processed, args.output_path)


if __name__ == "__main__":
    main()

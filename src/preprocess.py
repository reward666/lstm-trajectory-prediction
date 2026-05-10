import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_PATH = Path("/mnt/d/datasets/ngsim/trajectories-0750am-0805am.txt")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "ngsim_us101_0750_0805_processed.pkl"

FEET_TO_METER = 0.3048


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    columns = [
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

    df = pd.read_csv(
        path,
        sep=r"\s+",     # 空格分隔（关键）
        header=None,    # 没有表头（关键）
        names=columns,  # 手动指定列名（关键）
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


def main():
    print(f"Loading raw data from: {RAW_DATA_PATH}")

    df_raw = load_raw_data(RAW_DATA_PATH)
    print(f"Raw data shape: {df_raw.shape}")

    df_processed = clean_ngsim_data(df_raw)

    print_summary(df_processed)

    save_processed_data(df_processed, OUTPUT_PATH)


if __name__ == "__main__":
    main()
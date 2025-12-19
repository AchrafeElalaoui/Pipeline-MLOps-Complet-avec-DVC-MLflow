import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess retail demand data.")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--stores", default="data/stores.csv")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def to_int_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("TRUE").astype(int)


def load_data(train_path: str, features_path: str, stores_path: str) -> pd.DataFrame:
    train = pd.read_csv(train_path, parse_dates=["Date"])
    features = pd.read_csv(
        features_path,
        parse_dates=["Date"],
        na_values=["NA"],
        keep_default_na=True,
    )
    stores = pd.read_csv(stores_path)

    if "IsHoliday" in features.columns:
        features = features.drop(columns=["IsHoliday"])

    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on="Store", how="left")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    markdown_cols = [col for col in df.columns if col.startswith("MarkDown")]
    for col in markdown_cols:
        df[col] = df[col].fillna(0)

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.train, args.features, args.stores)
    df = df.sort_values("Date")
    df["IsHoliday"] = to_int_bool(df["IsHoliday"])
    df = add_time_features(df)
    df = fill_missing_values(df)
    df = df.drop(columns=["Date"])
    df = df.dropna(subset=["Weekly_Sales"])

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    df.to_csv(output_dir / "processed.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)

    print(f"Saved {len(train_df)} rows to {output_dir / 'train.csv'}")
    print(f"Saved {len(val_df)} rows to {output_dir / 'val.csv'}")


if __name__ == "__main__":
    main()

"""
Dataset loading, validation, cleaning, and reporting.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import normalize_label, get_logger

logger = get_logger("data_pipeline")


def load_and_clean(csv_path: str = None) -> pd.DataFrame:
    csv_path = csv_path or config.RAW_DATA_PATH
    df = pd.read_csv(csv_path)

    n_raw = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)

    n_missing = n_raw - len(df)
    empty_mask = df["text"].str.strip().eq("")
    df = df[~empty_mask].copy()
    n_empty = int(empty_mask.sum())

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    n_duplicates = n_before_dedup - len(df)

    df["label_norm"] = df["label"].map(normalize_label).astype(int)

    logger.info(
        f"Loaded {n_raw} rows -> {len(df)} after cleaning "
        f"(missing={n_missing}, empty={n_empty}, duplicates={n_duplicates})"
    )
    return df


def dataset_report(df: pd.DataFrame) -> dict:
    lengths = df["text"].str.len()
    word_counts = df["text"].str.split().str.len()
    report = {
        "total_samples": int(len(df)),
        "class_distribution": df["label_norm"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict() if "source" in df else {},
        "prompt_distribution_count": int(df["prompt_name"].nunique()) if "prompt_name" in df else None,
        "avg_text_length_chars": float(lengths.mean()),
        "min_text_length_chars": int(lengths.min()),
        "max_text_length_chars": int(lengths.max()),
        "median_text_length_chars": float(lengths.median()),
        "avg_word_count": float(word_counts.mean()),
        "min_word_count": int(word_counts.min()),
        "max_word_count": int(word_counts.max()),
    }
    return report


def save_report(report: dict, filename: str = "dataset_report.json"):
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = os.path.join(config.REPORTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Dataset report written to {path}")
    return path


if __name__ == "__main__":
    df = load_and_clean()
    report = dataset_report(df)
    save_report(report)
    print(json.dumps(report, indent=2, default=str))

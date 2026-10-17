"""
Lightweight experiment tracking (CSV-based, per the spec: "a CSV or
JSON-based implementation is sufficient... do not require MLflow").

Every training/tuning run appends one row. Columns are a superset across
runs -- missing columns for a given call are left blank.
"""
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

FIELDS = [
    "experiment_id", "timestamp", "dataset", "protocol", "model", "features",
    "hyperparameters", "train_rows", "valid_rows", "test_rows",
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "notes",
]


def log_experiment(**kwargs):
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = os.path.join(config.REPORTS_DIR, "experiments.csv")
    is_new = not os.path.exists(path)

    row = {k: kwargs.get(k, "") for k in FIELDS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    if not row["experiment_id"]:
        row["experiment_id"] = f"exp_{int(datetime.now().timestamp())}"

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row["experiment_id"]

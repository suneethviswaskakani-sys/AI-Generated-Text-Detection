"""
Robustness testing: run the trained ensemble against perturbed versions of
real test examples and report flip rate / accuracy under each perturbation.
See src/robustness.py for the perturbation functions and honesty notes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_and_clean
from src.robustness import run_robustness_suite
from predictor import Predictor
from src.utils import get_logger
from training.train import random_split

logger = get_logger("robustness_test")

N_SAMPLE = 200  # keep this small: each example needs 1 (original) + 5 (perturbations) predictions


def main():
    predictor = Predictor()
    df = load_and_clean()
    if config.TRAIN_SAMPLE_SIZE and len(df) > config.TRAIN_SAMPLE_SIZE:
        import pandas as pd
        frac = config.TRAIN_SAMPLE_SIZE / len(df)
        df = pd.concat([g.sample(frac=frac, random_state=config.RANDOM_STATE)
                         for _, g in df.groupby("label_norm")]).reset_index(drop=True)

    _, _, test_df = random_split(df)
    sample = test_df.sample(n=min(N_SAMPLE, len(test_df)), random_state=config.RANDOM_STATE)

    logger.info(f"Running robustness suite on {len(sample)} test examples...")
    report = run_robustness_suite(predictor, sample.text.tolist(), sample.label_norm.tolist())

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(config.REPORTS_DIR, "robustness_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

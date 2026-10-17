"""
Fine-tune and evaluate the optional transformer model (Model D), then
combine it with the already-trained word/char/structural models (from
training/train.py) into an "advanced ensemble".

This is a SEPARATE script from training/train.py on purpose: it must not
be able to break the working 3-model baseline pipeline (rule: don't delete
or destabilize working functionality to add an optional extra). Run
training/train.py first.

Requires: torch, transformers, and internet access to download the base
pretrained weights (distilbert-base-uncased by default) -- none of which
are available in the sandbox this project was authored in. This script has
NOT been executed end-to-end; run it yourself and inspect
data/reports/transformer_metrics.json for real numbers before citing any.

Usage:
    pip install torch transformers
    python training/train_transformer.py
"""
import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_and_clean
from src.features import vectorize_structural
from src.ensemble import compute_weights, combine, best_threshold
from src.transformer_model import TransformerTextClassifier, TransformerUnavailableError, is_available
from training.train import random_split  # reuse the same split logic/seed
from src.utils import get_logger

logger = get_logger("train_transformer")


def load_base_models():
    required = ["word_vectorizer.joblib", "char_vectorizer.joblib", "word_model.joblib",
                "char_model.joblib", "structural_scaler.joblib", "structural_model.joblib"]
    missing = [f for f in required if not os.path.exists(os.path.join(config.MODELS_DIR, f))]
    if missing:
        raise RuntimeError(
            f"Missing base model artifact(s) {missing}. Run training/train.py first -- "
            "this script only adds the transformer on top of that baseline."
        )
    return {
        "word": joblib.load(os.path.join(config.MODELS_DIR, "word_vectorizer.joblib")),
        "char": joblib.load(os.path.join(config.MODELS_DIR, "char_vectorizer.joblib")),
        "word_model": joblib.load(os.path.join(config.MODELS_DIR, "word_model.joblib")),
        "char_model": joblib.load(os.path.join(config.MODELS_DIR, "char_model.joblib")),
        "scaler": joblib.load(os.path.join(config.MODELS_DIR, "structural_scaler.joblib")),
        "structural_model": joblib.load(os.path.join(config.MODELS_DIR, "structural_model.joblib")),
    }


def base_predict(models, texts):
    Xw = models["word"].transform(texts)
    Xc = models["char"].transform(texts)
    sw = vectorize_structural(texts)
    A = models["scaler"].transform(sw)
    pw = models["word_model"].predict_proba(Xw)[:, 1]
    pc = models["char_model"].predict_proba(Xc)[:, 1]
    ps = models["structural_model"].predict_proba(A)[:, 1]
    return pw, pc, ps


def compute_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "n_samples": int(len(y)),
    }


def main():
    if not is_available():
        logger.error(
            "torch/transformers not installed -- cannot run. "
            "`pip install torch transformers` and rerun this script."
        )
        print(json.dumps({
            "status": "skipped",
            "reason": "torch/transformers not installed in this environment",
        }, indent=2))
        return

    base_models = load_base_models()
    df = load_and_clean()

    if config.TRANSFORMER_TRAIN_SAMPLE_SIZE and len(df) > config.TRANSFORMER_TRAIN_SAMPLE_SIZE:
        frac = config.TRANSFORMER_TRAIN_SAMPLE_SIZE / len(df)
        import pandas as pd
        df = pd.concat([
            g.sample(frac=frac, random_state=config.RANDOM_STATE)
            for _, g in df.groupby("label_norm")
        ]).reset_index(drop=True)
        logger.info(f"Subsampled to {len(df)} rows for transformer fine-tuning "
                    f"(fine-tuning is far more expensive per-row than the sklearn models).")

    # Reuse the exact same random-split logic/seed as train.py so the
    # transformer is compared on the same test set as the baseline models.
    train_df, valid_df, test_df = random_split(df)

    clf = TransformerTextClassifier()
    try:
        history = clf.fit(
            train_df.text.tolist(), train_df.label_norm.tolist(),
            valid_df.text.tolist(), valid_df.label_norm.tolist(),
        )
    except TransformerUnavailableError as e:
        logger.error(str(e))
        return

    clf.save()
    logger.info(f"Transformer saved to {config.TRANSFORMER_MODEL_DIR}")

    # Evaluate transformer alone
    pt_test = clf.predict_proba(test_df.text.tolist())
    yt = test_df.label_norm.values
    transformer_only_metrics = compute_metrics(yt, pt_test)

    # Combine with base models into a 4-model "advanced ensemble"
    pw_v, pc_v, ps_v = base_predict(base_models, valid_df.text.tolist())
    pt_v = clf.predict_proba(valid_df.text.tolist())
    yv = valid_df.label_norm.values
    weights = compute_weights(config.ENSEMBLE_WEIGHTING, yv, [pw_v, pc_v, ps_v, pt_v])
    pv = combine(weights, [pw_v, pc_v, ps_v, pt_v])
    threshold = best_threshold(yv, pv, config.THRESHOLD_SEARCH_RANGE)

    pw_t, pc_t, ps_t = base_predict(base_models, test_df.text.tolist())
    p_advanced = combine(weights, [pw_t, pc_t, ps_t, pt_test])
    advanced_ensemble_metrics = compute_metrics(yt, p_advanced, threshold)

    report = {
        "status": "completed",
        "training_history": history,
        "train_rows": int(len(train_df)), "valid_rows": int(len(valid_df)), "test_rows": int(len(test_df)),
        "transformer_only_test_metrics": transformer_only_metrics,
        "advanced_ensemble_weights": {
            "word_tfidf_lr": float(weights[0]), "char_tfidf_svm": float(weights[1]),
            "structural_gb": float(weights[2]), "transformer": float(weights[3]),
        },
        "advanced_ensemble_threshold": threshold,
        "advanced_ensemble_test_metrics": advanced_ensemble_metrics,
    }
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(config.REPORTS_DIR, "transformer_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(config.MODELS_DIR, "config.json")) as f:
        model_config = json.load(f)
    model_config["advanced_ensemble_weights"] = report["advanced_ensemble_weights"]
    model_config["advanced_ensemble_threshold"] = threshold
    model_config["transformer_available"] = True
    with open(os.path.join(config.MODELS_DIR, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

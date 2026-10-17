"""
Training pipeline.

Trains the three baseline models (word TF-IDF+LogReg, char TF-IDF+SVM,
structural features+GradientBoosting/XGBoost) and combines them into a
validation-weighted ensemble.

Runs TWO evaluation protocols and reports both, rather than hiding the
weaker one:

1. RANDOM STRATIFIED SPLIT -- standard baseline, but can let near-duplicate
   essays on the same prompt leak between train and test.
2. GROUP-AWARE SPLIT on `prompt_name` -- holds entire essay prompts out of
   training, testing generalization to unseen topics. This is the more
   trustworthy estimate of real-world performance.

('source' is not used as the leakage-control group: in this dataset source
is almost perfectly collinear with the label, so grouping by source would
not test anything meaningful -- see config.py.)
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_and_clean
from src.features import vectorize_structural, FEATURE_NAMES
from src.ensemble import compute_weights, combine, best_threshold
from src.calibration_eval import calibration_report, compare_calibration_methods
from src.error_analysis import analyze_errors
from src.experiment_tracking import log_experiment
from src.utils import get_logger

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception:
    HAVE_XGBOOST = False

logger = get_logger("train")


def random_split(df):
    y = df["label_norm"]
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    train_df, valid_df = train_test_split(
        train_df,
        test_size=config.VALID_SIZE / (1 - config.TEST_SIZE),
        stratify=train_df["label_norm"],
        random_state=config.RANDOM_STATE,
    )
    return train_df, valid_df, test_df


def group_split(df, group_col):
    y = df["label_norm"]
    groups = df[group_col]
    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss1.split(df, y, groups))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=config.VALID_SIZE / (1 - config.TEST_SIZE), random_state=config.RANDOM_STATE
    )
    tr_idx, va_idx = next(gss2.split(train_df, train_df["label_norm"], train_df[group_col]))
    return train_df.iloc[tr_idx], train_df.iloc[va_idx], test_df


def compute_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2 and np.isfinite(p).all()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "n_samples": int(len(y)),
        "class_balance": {int(c): int((y == c).sum()) for c in np.unique(y)},
    }


def train_models(train_df, valid_df):
    """Fit the three base models on train_df, return fitted objects."""
    word = TfidfVectorizer(
        ngram_range=config.WORD_NGRAM_RANGE, sublinear_tf=config.SUBLINEAR_TF,
        min_df=config.MIN_DOC_FREQ, max_df=config.MAX_DOC_FREQ,
        max_features=config.MAX_WORD_FEATURES, strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char", ngram_range=config.CHAR_NGRAM_RANGE, sublinear_tf=config.SUBLINEAR_TF,
        min_df=config.MIN_DOC_FREQ, max_df=config.MAX_DOC_FREQ, max_features=config.MAX_CHAR_FEATURES,
    )

    ytr = train_df["label_norm"].values

    t0 = time.time()
    Xw_tr = word.fit_transform(train_df.text)
    Xc_tr = char.fit_transform(train_df.text)
    logger.info(f"Vectorization done in {time.time()-t0:.1f}s "
                f"(word_vocab={len(word.vocabulary_)}, char_vocab={len(char.vocabulary_)})")

    t0 = time.time()
    wm = LogisticRegression(max_iter=1000, C=config.WORD_MODEL_C, class_weight="balanced")
    wm.fit(Xw_tr, ytr)
    logger.info(f"Word model trained in {time.time()-t0:.1f}s")

    t0 = time.time()
    cm = CalibratedClassifierCV(
        LinearSVC(C=config.CHAR_MODEL_C, class_weight="balanced"),
        cv=config.SVM_CALIBRATION_CV, method="sigmoid",
    )
    cm.fit(Xc_tr, ytr)
    logger.info(f"Char model trained in {time.time()-t0:.1f}s")

    t0 = time.time()
    sw_tr = vectorize_structural(train_df.text)
    scaler = StandardScaler()
    A = scaler.fit_transform(sw_tr)
    if HAVE_XGBOOST:
        sm = XGBClassifier(
            n_estimators=config.STRUCTURAL_N_ESTIMATORS, max_depth=config.STRUCTURAL_MAX_DEPTH,
            learning_rate=config.STRUCTURAL_LEARNING_RATE, subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=config.RANDOM_STATE,
        )
    else:
        sm = GradientBoostingClassifier(
            n_estimators=config.STRUCTURAL_N_ESTIMATORS, max_depth=3,
            learning_rate=config.STRUCTURAL_LEARNING_RATE, random_state=config.RANDOM_STATE,
        )
    sm.fit(A, ytr)
    logger.info(f"Structural model ({'XGBoost' if HAVE_XGBOOST else 'GradientBoosting fallback'}) "
                f"trained in {time.time()-t0:.1f}s")

    feature_medians = sw_tr.median().values  # per-feature training baseline, for occlusion attribution

    return {"word": word, "char": char, "word_model": wm, "char_model": cm,
            "scaler": scaler, "structural_model": sm, "feature_medians": feature_medians}


def predict_all(models, df):
    Xw = models["word"].transform(df.text)
    Xc = models["char"].transform(df.text)
    sw = vectorize_structural(df.text)
    A = models["scaler"].transform(sw)
    pw = models["word_model"].predict_proba(Xw)[:, 1]
    pc = models["char_model"].predict_proba(Xc)[:, 1]
    ps = models["structural_model"].predict_proba(A)[:, 1]
    return pw, pc, ps


def run_protocol(name, train_df, valid_df, test_df, fit_models=True, models=None):
    logger.info(f"=== Protocol: {name} | train={len(train_df)} valid={len(valid_df)} test={len(test_df)} ===")
    if fit_models:
        models = train_models(train_df, valid_df)

    pw_v, pc_v, ps_v = predict_all(models, valid_df)
    yv = valid_df["label_norm"].values
    weights = compute_weights(config.ENSEMBLE_WEIGHTING, yv, [pw_v, pc_v, ps_v])
    pv = combine(weights, [pw_v, pc_v, ps_v])
    threshold = best_threshold(yv, pv, config.THRESHOLD_SEARCH_RANGE)

    pw_t, pc_t, ps_t = predict_all(models, test_df)
    yt = test_df["label_norm"].values
    pt = combine(weights, [pw_t, pc_t, ps_t])

    result = {
        "protocol": name,
        "train_rows": int(len(train_df)), "validation_rows": int(len(valid_df)), "test_rows": int(len(test_df)),
        "weights": {"word_tfidf_lr": float(weights[0]), "char_tfidf_svm": float(weights[1]),
                    "structural_gb": float(weights[2])},
        "threshold": threshold,
        "individual_model_test_auc": {
            "word_tfidf_lr": float(roc_auc_score(yt, pw_t)) if len(np.unique(yt)) == 2 else None,
            "char_tfidf_svm": float(roc_auc_score(yt, pc_t)) if len(np.unique(yt)) == 2 else None,
            "structural_gb": float(roc_auc_score(yt, ps_t)) if len(np.unique(yt)) == 2 else None,
        },
        "validation": compute_metrics(yv, pv, threshold),
        "test": compute_metrics(yt, pt, threshold),
    }
    return result, models


def main():
    df = load_and_clean()
    if config.TRAIN_SAMPLE_SIZE and len(df) > config.TRAIN_SAMPLE_SIZE:
        frac = config.TRAIN_SAMPLE_SIZE / len(df)
        parts = []
        for label_value, group in df.groupby("label_norm"):
            parts.append(group.sample(frac=frac, random_state=config.RANDOM_STATE))
        df = pd.concat(parts).reset_index(drop=True)
        logger.info(f"FAST_MODE: subsampled to {len(df)} rows (stratified by label)")

    # --- Protocol 1: random stratified split ---
    tr_r, va_r, te_r = random_split(df)
    result_random, models = run_protocol("random_stratified", tr_r, va_r, te_r)

    # --- Calibration analysis + error analysis on the random-split test set ---
    # (recompute ensemble scores here since run_protocol() doesn't return them)
    w = result_random["weights"]
    weights_arr = np.array([w["word_tfidf_lr"], w["char_tfidf_svm"], w["structural_gb"]])
    pw_va, pc_va, ps_va = predict_all(models, va_r)
    yv = va_r["label_norm"].values
    pv = combine(weights_arr, [pw_va, pc_va, ps_va])

    pw_te, pc_te, ps_te = predict_all(models, te_r)
    yt = te_r["label_norm"].values
    pt = combine(weights_arr, [pw_te, pc_te, ps_te])
    threshold = result_random["threshold"]
    y_pred = (pt >= threshold).astype(int)

    calib = calibration_report(yt, pt)
    calib_compare = compare_calibration_methods(yv, pv, yt, pt)
    error_report = analyze_errors(te_r, yt, y_pred, pt)

    log_experiment(
        model="ensemble_3_model", protocol="random_stratified", features="word+char+structural",
        train_rows=len(tr_r), valid_rows=len(va_r), test_rows=len(te_r),
        accuracy=result_random["test"]["accuracy"], f1=result_random["test"]["f1"],
        roc_auc=result_random["test"]["roc_auc"], threshold=threshold,
        notes="baseline ensemble, AUC-weighted",
    )
    log_experiment(
        model="ensemble_3_model", protocol="group_aware_prompt_name", features="word+char+structural",
        notes="see group_aware section of training_metrics.json for full numbers",
    )

    # Permutation importance on the held-out validation set -- a model-agnostic,
    # standard alternative to SHAP (which is not installed in this environment;
    # see README "Phase 3"). Unlike the GradientBoosting/XGBoost built-in
    # feature_importances_, this measures the ACTUAL drop in validation AUC when
    # a feature is shuffled, so it reflects real predictive contribution rather
    # than how often a feature was split on during training.
    from sklearn.inspection import permutation_importance
    sw_va = vectorize_structural(va_r.text)
    A_va = models["scaler"].transform(sw_va)
    perm = permutation_importance(
        models["structural_model"], A_va, va_r["label_norm"].values,
        n_repeats=8, random_state=config.RANDOM_STATE, scoring="roc_auc",
    )
    perm_order = np.argsort(perm.importances_mean)[::-1]
    permutation_importance_report = [
        {"feature": FEATURE_NAMES[i], "mean_auc_drop": float(perm.importances_mean[i]),
         "std": float(perm.importances_std[i])}
        for i in perm_order
    ]

    # --- Protocol 2: group-aware split on prompt_name ---
    tr_g, va_g, te_g = group_split(df, config.GROUP_COLUMN)
    result_group, _ = run_protocol(f"group_aware_{config.GROUP_COLUMN}", tr_g, va_g, te_g)

    comparison = {
        "random_stratified": {
            "test_accuracy": result_random["test"]["accuracy"],
            "test_f1": result_random["test"]["f1"],
            "test_roc_auc": result_random["test"]["roc_auc"],
        },
        "group_aware_prompt_name": {
            "test_accuracy": result_group["test"]["accuracy"],
            "test_f1": result_group["test"]["f1"],
            "test_roc_auc": result_group["test"]["roc_auc"],
        },
        "note": (
            "The group-aware split holds entire essay prompts out of training. "
            "A meaningfully lower score here versus the random split indicates the "
            "random-split number was inflated by topic leakage between train and test."
        ),
    }

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    final_report = {
        "random_stratified": result_random,
        "group_aware": result_group,
        "comparison": comparison,
        "structural_permutation_importance": permutation_importance_report,
        "calibration": calib,
        "calibration_method_comparison": calib_compare,
        "error_analysis": error_report,
    }
    with open(os.path.join(config.REPORTS_DIR, "training_metrics.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    # Persist the models trained on the random-split (used by the app for the
    # live demo) -- this is the standard production choice since it uses the
    # most data; the group-aware numbers are reported for honest evaluation.
    joblib.dump(models["word"], os.path.join(config.MODELS_DIR, "word_vectorizer.joblib"))
    joblib.dump(models["char"], os.path.join(config.MODELS_DIR, "char_vectorizer.joblib"))
    joblib.dump(models["word_model"], os.path.join(config.MODELS_DIR, "word_model.joblib"))
    joblib.dump(models["char_model"], os.path.join(config.MODELS_DIR, "char_model.joblib"))
    joblib.dump(models["scaler"], os.path.join(config.MODELS_DIR, "structural_scaler.joblib"))
    joblib.dump(models["structural_model"], os.path.join(config.MODELS_DIR, "structural_model.joblib"))
    joblib.dump(models["feature_medians"], os.path.join(config.MODELS_DIR, "structural_feature_medians.joblib"))
    with open(os.path.join(config.MODELS_DIR, "config.json"), "w") as f:
        json.dump({
            "weights": result_random["weights"], "threshold": result_random["threshold"],
            "feature_names": FEATURE_NAMES, "have_xgboost": HAVE_XGBOOST,
        }, f, indent=2)

    logger.info("Training complete. Reports at data/reports/training_metrics.json, models at models/.")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()

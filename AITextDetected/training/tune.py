"""
Optional hyperparameter tuning, per spec: "Do not perform extremely
expensive searches by default... FAST_MODE = True for normal laptop
execution."

Tunes the word model (LogisticRegression C) and structural model
(n_estimators, max_depth, learning_rate) using RandomizedSearchCV on a
subsample, logs every trial to the experiment tracker, and reports the
best params found -- without touching the test set.
"""
import os
import sys
import time

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_and_clean
from src.features import vectorize_structural
from src.experiment_tracking import log_experiment
from src.utils import get_logger

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception:
    HAVE_XGBOOST = False

logger = get_logger("tune")

N_ITER = 6 if config.FAST_MODE else 20
CV_FOLDS = 3
TUNE_SAMPLE_SIZE = 6000 if config.FAST_MODE else None


def tune_word_model(df):
    y = df["label_norm"].values
    word = TfidfVectorizer(
        ngram_range=config.WORD_NGRAM_RANGE, sublinear_tf=True,
        min_df=config.MIN_DOC_FREQ, max_df=config.MAX_DOC_FREQ, max_features=config.MAX_WORD_FEATURES,
    )
    X = word.fit_transform(df.text)

    param_dist = {"C": uniform(0.1, 10.0), "class_weight": [None, "balanced"]}
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=500), param_dist, n_iter=N_ITER,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1", random_state=config.RANDOM_STATE, n_jobs=1,
    )
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0
    logger.info(f"Word model tuning done in {elapsed:.1f}s. Best F1={search.best_score_:.4f}, "
                f"params={search.best_params_}")

    for i in range(len(search.cv_results_["params"])):
        log_experiment(
            model="word_tfidf_lr", features="word_tfidf",
            hyperparameters=str(search.cv_results_["params"][i]),
            f1=round(float(search.cv_results_["mean_test_score"][i]), 4),
            notes=f"randomized_search_trial_{i}", train_rows=len(df),
        )
    return search.best_params_, search.best_score_


def tune_structural_model(df):
    y = df["label_norm"].values
    struct = vectorize_structural(df.text)

    param_dist = {
        "n_estimators": randint(50, 200),
        "max_depth": randint(2, 6),
        "learning_rate": uniform(0.02, 0.18),
    }
    base = (XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_STATE)
            if HAVE_XGBOOST else GradientBoostingClassifier(random_state=config.RANDOM_STATE))
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_ITER,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1", random_state=config.RANDOM_STATE, n_jobs=1,
    )
    t0 = time.time()
    search.fit(struct, y)
    elapsed = time.time() - t0
    logger.info(f"Structural model tuning done in {elapsed:.1f}s. Best F1={search.best_score_:.4f}, "
                f"params={search.best_params_}")

    for i in range(len(search.cv_results_["params"])):
        log_experiment(
            model="structural_gb" if not HAVE_XGBOOST else "structural_xgb",
            features="stylometric",
            hyperparameters=str(search.cv_results_["params"][i]),
            f1=round(float(search.cv_results_["mean_test_score"][i]), 4),
            notes=f"randomized_search_trial_{i}", train_rows=len(df),
        )
    return search.best_params_, search.best_score_


def main():
    df = load_and_clean()
    if TUNE_SAMPLE_SIZE and len(df) > TUNE_SAMPLE_SIZE:
        import pandas as pd
        frac = TUNE_SAMPLE_SIZE / len(df)
        df = pd.concat([g.sample(frac=frac, random_state=config.RANDOM_STATE)
                         for _, g in df.groupby("label_norm")]).reset_index(drop=True)
        logger.info(f"Tuning on a {len(df)}-row subsample (FAST_MODE).")

    word_params, word_f1 = tune_word_model(df)
    struct_params, struct_f1 = tune_structural_model(df)

    result = {
        "word_model": {"best_params": word_params, "cv_f1": word_f1},
        "structural_model": {"best_params": struct_params, "cv_f1": struct_f1},
        "n_iter_per_model": N_ITER, "cv_folds": CV_FOLDS, "tuned_on_rows": len(df),
    }
    import json
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(config.REPORTS_DIR, "tuning_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

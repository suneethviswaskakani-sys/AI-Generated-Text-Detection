"""
Ensemble weighting strategies, computed from validation data only
(never from test data -- avoids leaking test information into the model).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def equal_weights(n_models: int) -> np.ndarray:
    return np.full(n_models, 1.0 / n_models)


def auc_weights(y_valid, probas: list) -> np.ndarray:
    aucs = np.array([roc_auc_score(y_valid, p) for p in probas], dtype=float)
    aucs = np.nan_to_num(aucs, nan=0.5, posinf=1.0, neginf=0.5)
    # Shift so a model at chance level (AUC 0.5) contributes ~0, floor to avoid
    # a single failed model zeroing itself out completely.
    w = np.maximum(aucs - 0.5, 0.01)
    return w / w.sum()


def f1_weights(y_valid, probas: list, threshold: float = 0.5) -> np.ndarray:
    f1s = np.array(
        [f1_score(y_valid, (p >= threshold).astype(int), zero_division=0) for p in probas],
        dtype=float,
    )
    w = np.maximum(f1s, 0.01)
    return w / w.sum()


def compute_weights(method: str, y_valid, probas: list) -> np.ndarray:
    if method == "equal":
        return equal_weights(len(probas))
    if method == "f1":
        return f1_weights(y_valid, probas)
    return auc_weights(y_valid, probas)  # default: "auc"


def combine(weights: np.ndarray, probas: list) -> np.ndarray:
    out = np.zeros_like(probas[0], dtype=float)
    for w, p in zip(weights, probas):
        out += w * p
    return out


def best_threshold(y_valid, p_valid, search_range=(0.30, 0.70, 81)) -> float:
    lo, hi, n = search_range
    thresholds = np.linspace(lo, hi, n)
    scores = [f1_score(y_valid, (p_valid >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])

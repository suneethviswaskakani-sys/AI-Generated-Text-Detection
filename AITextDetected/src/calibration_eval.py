"""
Probability calibration analysis: is a 0.8 "prediction score" actually
right ~80% of the time? Reports a calibration curve (bins) and Brier score,
and supports fitting Platt/isotonic calibration on top of raw ensemble
scores (per spec section 9 -- "if probability calibration is implemented,
support Platt scaling, isotonic calibration, calibration curves, Brier score").
"""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibration_report(y_true, y_prob, n_bins: int = 10) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "brier_score": brier,
        "bins": [
            {"mean_predicted": float(mp), "fraction_positive": float(fp)}
            for mp, fp in zip(mean_pred, frac_pos)
        ],
        "note": (
            "A well-calibrated model has fraction_positive close to mean_predicted in "
            "each bin. Brier score (mean squared error of the probability) closer to 0 is better; "
            "0.25 is the score of a coin-flip classifier on a balanced dataset."
        ),
    }


def fit_platt(y_valid, p_valid) -> LogisticRegression:
    """Platt scaling: fit a 1D logistic regression from raw score -> calibrated probability."""
    lr = LogisticRegression()
    lr.fit(p_valid.reshape(-1, 1), y_valid)
    return lr


def fit_isotonic(y_valid, p_valid) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_valid, y_valid)
    return iso


def compare_calibration_methods(y_valid, p_valid, y_test, p_test) -> dict:
    """Fit both calibrators on validation data, evaluate Brier score on test data,
    alongside the raw (uncalibrated) ensemble score, so the best method can be chosen
    using validation-fit / test-evaluated data (never touching test during fitting)."""
    platt = fit_platt(y_valid, p_valid)
    iso = fit_isotonic(y_valid, p_valid)

    p_test_platt = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]
    p_test_iso = iso.predict(p_test)

    return {
        "raw_brier": float(brier_score_loss(y_test, p_test)),
        "platt_brier": float(brier_score_loss(y_test, p_test_platt)),
        "isotonic_brier": float(brier_score_loss(y_test, p_test_iso)),
    }

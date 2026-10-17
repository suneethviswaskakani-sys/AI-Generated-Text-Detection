"""
Error analysis: find false positives, false negatives, and uncertain
predictions on the test set, and summarize what distinguishes them from
correct predictions using the stylometric features (which are directly
interpretable, unlike TF-IDF dimensions).
"""
import numpy as np
import pandas as pd

from src.features import vectorize_structural, FEATURE_NAMES


def analyze_errors(test_df, y_true, y_pred, p_final, top_n: int = 5) -> dict:
    test_df = test_df.reset_index(drop=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_final = np.asarray(p_final)

    fp_mask = (y_true == 0) & (y_pred == 1)   # human text called AI
    fn_mask = (y_true == 1) & (y_pred == 0)   # AI text called human
    uncertain_mask = np.abs(p_final - 0.5) < 0.05

    def _examples(mask, n):
        idx = np.where(mask)[0]
        # pick the most confidently wrong examples -- most informative for error analysis
        idx = idx[np.argsort(-np.abs(p_final[idx] - 0.5))][:n]
        return [
            {
                "text_preview": test_df.text.iloc[i][:200],
                "true_label": int(y_true[i]),
                "predicted_score": float(p_final[i]),
                "source": test_df.source.iloc[i] if "source" in test_df.columns else None,
                "prompt_name": test_df.prompt_name.iloc[i] if "prompt_name" in test_df.columns else None,
            }
            for i in idx
        ]

    # Compare average stylometric features of errors vs. correct predictions
    struct = vectorize_structural(test_df.text)
    correct_mask = y_true == y_pred
    feature_comparison = []
    if fp_mask.sum() > 0 or fn_mask.sum() > 0:
        for feat in FEATURE_NAMES:
            correct_mean = float(struct.loc[correct_mask, feat].mean())
            fp_mean = float(struct.loc[fp_mask, feat].mean()) if fp_mask.sum() > 0 else None
            fn_mean = float(struct.loc[fn_mask, feat].mean()) if fn_mask.sum() > 0 else None
            feature_comparison.append({
                "feature": feat, "correct_mean": correct_mean,
                "false_positive_mean": fp_mean, "false_negative_mean": fn_mean,
            })

    return {
        "n_test": int(len(test_df)),
        "n_false_positives": int(fp_mask.sum()),
        "n_false_negatives": int(fn_mask.sum()),
        "n_uncertain": int(uncertain_mask.sum()),
        "false_positive_rate": float(fp_mask.sum() / max(1, (y_true == 0).sum())),
        "false_negative_rate": float(fn_mask.sum() / max(1, (y_true == 1).sum())),
        "false_positive_examples": _examples(fp_mask, top_n),
        "false_negative_examples": _examples(fn_mask, top_n),
        "feature_comparison": feature_comparison,
    }

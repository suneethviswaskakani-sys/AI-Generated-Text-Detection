"""
Multi-factor Reliability Assessment Engine.
Evaluates prediction trustworthiness across text length, extraction quality,
cross-model consensus, and decision boundary margins.
"""
import os
import sys
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def assess_reliability(
    word_count: int,
    model_agreement_std: Optional[float] = None,
    extraction_quality: str = "HIGH",
    decision_margin: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute a composite reliability score (High, Medium, Low, Insufficient).
    Combines:
    - Text length (shorter texts have higher stylometric variance)
    - Document extraction quality (OCR status, blank pages)
    - Ensemble agreement (low standard deviation across independent sub-models)
    - Proximity to the decision boundary
    """
    t = config.RELIABILITY_THRESHOLDS
    reasons = []
    penalty = 0.0

    # 1. Word count assessment
    if word_count < t["insufficient"]:
        return {
            "tier": "insufficient",
            "score": 0.1,
            "label": "Insufficient Text",
            "message": (
                f"Document contains only {word_count} words. Below {t['insufficient']} words, "
                "statistical NLP features lack sufficient sample density."
            ),
            "usable": False,
            "factors": {"length": "insufficient", "agreement": "unknown", "extraction": extraction_quality},
        }

    if word_count < t["low"]:
        penalty += 0.35
        reasons.append(f"Short document length ({word_count} words) provides a weaker statistical signal.")
        length_tier = "low"
    elif word_count < t["moderate"]:
        penalty += 0.15
        reasons.append(f"Moderate document length ({word_count} words). Best reliability is achieved with 250+ words.")
        length_tier = "moderate"
    else:
        length_tier = "optimal"

    # 2. Extraction quality assessment
    if extraction_quality == "POOR / OCR NEEDED":
        penalty += 0.50
        reasons.append("Document extraction quality is poor (possible scanned images without text layer).")
    elif extraction_quality == "LOW":
        penalty += 0.25
        reasons.append("Document extraction yielded sparse or degraded text.")

    # 3. Model consensus assessment
    agreement_tier = "high"
    if model_agreement_std is not None:
        if model_agreement_std >= 0.28:
            penalty += 0.30
            agreement_tier = "divergent"
            reasons.append(f"High disagreement among models (std={model_agreement_std:.2f}); features conflict.")
        elif model_agreement_std >= 0.18:
            penalty += 0.15
            agreement_tier = "moderate"
            reasons.append(f"Moderate variance across model representations (std={model_agreement_std:.2f}).")

    # 4. Proximity to decision boundary
    margin_tier = "wide"
    if decision_margin is not None:
        if decision_margin < 0.08:
            penalty += 0.20
            margin_tier = "borderline"
            reasons.append("Prediction falls immediately adjacent to the decision threshold (borderline).")
        elif decision_margin < 0.16:
            penalty += 0.10
            margin_tier = "narrow"

    # Final composite score in [0.0, 1.0]
    base_score = 1.0 - min(0.85, penalty)

    if base_score >= 0.80:
        tier = "normal"
        label = "High Reliability"
        summary = "Text length, model consensus, and document quality meet enterprise criteria."
    elif base_score >= 0.55:
        tier = "moderate"
        label = "Medium Reliability"
        summary = "Result is informative; " + " ".join(reasons)
    else:
        tier = "low"
        label = "Low Reliability"
        summary = "Treat with caution; " + " ".join(reasons)

    return {
        "tier": tier,
        "score": round(base_score, 2),
        "label": label,
        "message": summary,
        "usable": True,
        "reasons": reasons,
        "factors": {
            "length": length_tier,
            "extraction": extraction_quality,
            "agreement": agreement_tier,
            "margin": margin_tier,
        },
    }

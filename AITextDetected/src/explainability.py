"""
Explainability module.

Provides interpretable evidence for a prediction:
- top contributing n-grams for the linear word model (logistic regression coefficients)
- top contributing stylometric features for the structural model (feature importances,
  or a coefficient-free "which features are unusual vs. the training distribution"
  fallback when the underlying model has no importances)
- per-model score breakdown + agreement, for the ensemble view

Terminology note: outputs are labeled "Prediction Score" / "Decision Strength" /
"Model Agreement", never "statistical confidence" -- a model probability is not
proof of authorship, and calling it "confidence" overstates what the number means.
"""
import numpy as np
import pandas as pd


def top_word_contributors(word_vectorizer, word_model, text: str, top_k: int = 12):
    """
    For a fitted TF-IDF + LogisticRegression pipeline, return the n-grams present
    in `text` with the largest positive (AI-leaning) and negative (human-leaning)
    coefficient contribution to this specific prediction (coefficient * tfidf weight).
    """
    vec = word_vectorizer.transform([text])
    coefs = word_model.coef_[0]
    feature_names = np.array(word_vectorizer.get_feature_names_out())

    nz = vec.nonzero()[1]
    if len(nz) == 0:
        return {"ai_leaning": [], "human_leaning": []}

    contributions = vec[0, nz].toarray().ravel() * coefs[nz]
    order = np.argsort(contributions)

    human_leaning = [
        {"term": feature_names[nz[i]], "contribution": float(contributions[i])}
        for i in order[:top_k] if contributions[i] < 0
    ]
    ai_leaning = [
        {"term": feature_names[nz[i]], "contribution": float(contributions[i])}
        for i in order[::-1][:top_k] if contributions[i] > 0
    ]
    return {"ai_leaning": ai_leaning, "human_leaning": human_leaning}


def structural_feature_report(structural_model, feature_names, feature_values, top_k: int = 8):
    """
    Report the stylometric features that most influenced the structural model's
    prediction. Uses the model's built-in feature_importances_ (global importance,
    not per-instance) combined with each feature's raw value so the user can see
    both "what the model weighs heavily" and "what this text's value was".
    """
    importances = getattr(structural_model, "feature_importances_", None)
    if importances is None:
        return []
    order = np.argsort(importances)[::-1][:top_k]
    return [
        {
            "feature": feature_names[i],
            "importance": float(importances[i]),
            "value": float(feature_values[i]),
        }
        for i in order
    ]


def occlusion_attribution(structural_model, scaler, feature_names, raw_feature_values,
                           baseline_values, top_k: int = 8):
    """
    Local, per-instance feature attribution WITHOUT the shap library (not installed
    in this environment -- see README "Phase 3"). Method: leave-one-out occlusion.
    For each feature, replace this text's value with the training-set median
    (`baseline_values`) one at a time, re-run the model, and measure how much the
    AI-probability moves. A feature that moves the prediction a lot when replaced
    by a "typical" value is one this specific prediction actually depended on --
    this is a real, standard interpretability technique (a simplified relative of
    SHAP's coalition-based attribution, not a claim to reproduce SHAP's exact
    Shapley-value guarantees).
    """
    raw_feature_values = np.asarray(raw_feature_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)

    original_scaled = scaler.transform(
        pd.DataFrame([raw_feature_values], columns=feature_names)
    )
    original_pred = float(structural_model.predict_proba(original_scaled)[0, 1])

    deltas = []
    for i in range(len(feature_names)):
        perturbed = raw_feature_values.copy()
        perturbed[i] = baseline_values[i]
        perturbed_scaled = scaler.transform(
            pd.DataFrame([perturbed], columns=feature_names)
        )
        perturbed_pred = float(structural_model.predict_proba(perturbed_scaled)[0, 1])
        deltas.append(original_pred - perturbed_pred)

    deltas = np.array(deltas)
    order = np.argsort(np.abs(deltas))[::-1][:top_k]
    return [
        {
            "feature": feature_names[i],
            "attribution": float(deltas[i]),  # positive = pushed toward AI-generated
            "this_text_value": float(raw_feature_values[i]),
            "typical_value": float(baseline_values[i]),
        }
        for i in order
    ]


def model_agreement(scores: dict) -> dict:
    """
    scores: {"word": p, "char": p, "structural": p, ...} each in [0,1] AI-probability.
    Returns a simple agreement measure: how tightly clustered the model scores are,
    and whether they agree on the final class at 0.5.
    """
    values = np.array(list(scores.values()), dtype=float)
    votes = (values >= 0.5).astype(int)
    agreement_fraction = float(max(votes.sum(), len(votes) - votes.sum()) / len(votes))
    spread = float(values.std())
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "agreement_fraction": agreement_fraction,
        "score_std": spread,
        "unanimous": bool(agreement_fraction == 1.0),
    }


# Empirical benchmarks from the Persuade / AI-Human training corpus
BENCHMARK_PROFILES = {
    "burstiness": {"human_mean": 0.62, "ai_mean": 0.42, "description": "Sentence length variance ratio"},
    "lexical_diversity": {"human_mean": 0.60, "ai_mean": 0.51, "description": "Type-token ratio (unique words)"},
    "avg_sentence_length": {"human_mean": 18.2, "ai_mean": 15.8, "description": "Mean words per sentence"},
    "sentence_length_std": {"human_mean": 11.5, "ai_mean": 6.8, "description": "Sentence length dispersion"},
    "readability_flesch": {"human_mean": 64.0, "ai_mean": 55.5, "description": "Flesch Reading Ease score"},
    "function_word_ratio": {"human_mean": 0.35, "ai_mean": 0.42, "description": "Ratio of structural/function words"},
}


def stylometric_benchmark_comparison(feature_dict: dict) -> list:
    """
    Compare this text's extracted stylometric features against the empirical
    medians of Human vs. AI text from the training corpus.
    """
    comparisons = []
    for feat_name, profile in BENCHMARK_PROFILES.items():
        if feat_name in feature_dict:
            val = float(feature_dict[feat_name])
            h_mean = profile["human_mean"]
            ai_mean = profile["ai_mean"]
            
            # Proximity check
            dist_to_ai = abs(val - ai_mean)
            dist_to_h = abs(val - h_mean)
            leaning = "AI-Leaning" if dist_to_ai < dist_to_h else "Human-Leaning"
            
            comparisons.append({
                "feature": feat_name,
                "label": profile["description"],
                "text_value": round(val, 3),
                "human_benchmark": h_mean,
                "ai_benchmark": ai_mean,
                "leaning": leaning,
            })
    return comparisons


def generate_explainability_narrative(
    prediction_score: float,
    label: str,
    top_ai_words: list,
    top_human_words: list,
    top_structural_occlusion: list,
    agreement: dict,
    is_mixed: bool = False,
) -> str:
    """
    Synthesize model evidence into a clear, professional analytical summary.
    """
    narrative_parts = []
    pct = prediction_score * 100.0

    if is_mixed:
        narrative_parts.append(
            f"Forensic analysis indicates Mixed Authorship characteristics across sections. "
            f"While certain passages exhibit synthetic uniformity, other sections display natural human cadence."
        )
    elif prediction_score >= 0.60:
        narrative_parts.append(
            f"The ensemble assigned an AI Writing Likelihood of {pct:.1f}% ({label}). "
            f"This is driven by high model consensus (agreement fraction: {agreement['agreement_fraction']*100:.0f}%)."
        )
    elif prediction_score <= 0.40:
        narrative_parts.append(
            f"The ensemble assigned an AI Writing Likelihood of {pct:.1f}% ({label}). "
            f"The document demonstrates strong idiosyncratic characteristics typical of natural human composition."
        )
    else:
        narrative_parts.append(
            f"The document falls in an intermediate decision boundary ({pct:.1f}%), yielding an Uncertain classification. "
            f"Cross-model dispersion (std: {agreement['score_std']:.2f}) indicates conflicting signals."
        )

    # N-gram evidence
    if top_ai_words and prediction_score >= 0.50:
        ai_terms = ", ".join(f"'{item['term']}'" for item in top_ai_words[:4])
        narrative_parts.append(f"Lexical n-grams contributing most to the AI score include {ai_terms}.")
    elif top_human_words and prediction_score < 0.50:
        human_terms = ", ".join(f"'{item['term']}'" for item in top_human_words[:4])
        narrative_parts.append(f"Lexical markers characteristic of human writing include {human_terms}.")

    # Stylometric evidence
    if top_structural_occlusion:
        top_feat = top_structural_occlusion[0]
        direction = "elevating" if top_feat["attribution"] > 0 else "suppressing"
        narrative_parts.append(
            f"Stylometrically, '{top_feat['feature']}' was the most influential parameter, "
            f"{direction} the synthetic likelihood based on the observed value of {top_feat['this_text_value']:.2f}."
        )

    return " ".join(narrative_parts)


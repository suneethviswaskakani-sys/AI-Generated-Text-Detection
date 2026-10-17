"""
Chunking, sliding-window analysis, and mixed authorship detection engine.
Enables accurate forensic evaluation of real-world documents (PDFs, research papers,
multi-page essays) by preventing out-of-distribution feature distortion and
detecting partial AI generation.
"""
import math
import re
from typing import List, Dict, Any, Tuple
import numpy as np


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with boundary preservation."""
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []
    return sentences


def create_chunks(
    text: str,
    target_words: int = 280,
    overlap_words: int = 50,
    min_chunk_words: int = 70,
) -> List[Dict[str, Any]]:
    """
    Split text into sentence-aligned sliding window chunks.
    Ensures that chunks respect sentence boundaries while staying close
    to the training distribution (~250-400 words).
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences: List[str] = []
    current_word_count = 0
    char_offset = 0

    # If the entire text is shorter than target_words, return as single chunk
    total_words = len(text.split())
    if total_words <= target_words + 50:
        return [{
            "chunk_id": 1,
            "text": text.strip(),
            "word_count": total_words,
            "start_word": 0,
            "end_word": total_words,
            "start_char": 0,
            "end_char": len(text),
        }]

    i = 0
    total_sentences = len(sentences)
    chunk_id = 1
    running_word_idx = 0

    while i < total_sentences:
        current_chunk_sents = []
        words_in_chunk = 0
        start_sentence_idx = i

        while i < total_sentences and (words_in_chunk < target_words or words_in_chunk < min_chunk_words):
            sent = sentences[i]
            w_count = len(sent.split())
            current_chunk_sents.append(sent)
            words_in_chunk += w_count
            i += 1

        chunk_text = " ".join(current_chunk_sents).strip()
        if len(chunk_text.split()) >= min_chunk_words:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "start_sentence_idx": start_sentence_idx,
                "end_sentence_idx": i - 1,
                "start_word": running_word_idx,
                "end_word": running_word_idx + len(chunk_text.split()),
            })
            chunk_id += 1
            running_word_idx += max(1, len(chunk_text.split()) - overlap_words)

        # Backtrack by overlap sentences if possible and not at end
        if i < total_sentences:
            overlap_count = 0
            backtrack_steps = 0
            for s in reversed(current_chunk_sents):
                s_words = len(s.split())
                if overlap_count + s_words <= overlap_words and backtrack_steps < (len(current_chunk_sents) - 1):
                    overlap_count += s_words
                    backtrack_steps += 1
                else:
                    break
            i = max(start_sentence_idx + 1, i - backtrack_steps)

    return chunks


def detect_mixed_authorship(
    chunk_scores: List[float],
    threshold_low: float = 0.40,
    threshold_high: float = 0.60,
    variance_threshold: float = 0.045,
) -> Dict[str, Any]:
    """
    Analyze chunk-level AI probabilities to detect mixed authorship
    (e.g., student-written essay with AI-generated body paragraphs).
    """
    if len(chunk_scores) < 2:
        return {
            "is_mixed": False,
            "mixed_confidence": 0.0,
            "score_variance": 0.0,
            "score_std": 0.0,
            "ai_ratio": 1.0 if chunk_scores and chunk_scores[0] >= 0.5 else 0.0,
            "transition_count": 0,
            "assessment": "Insufficient chunks to assess mixed authorship.",
        }

    arr = np.array(chunk_scores, dtype=float)
    var = float(np.var(arr))
    std = float(np.std(arr))

    ai_mask = arr >= threshold_high
    human_mask = arr <= threshold_low
    ai_count = int(ai_mask.sum())
    human_count = int(human_mask.sum())
    n = len(chunk_scores)

    # Transitions between human and AI
    labels = []
    for s in arr:
        if s >= threshold_high:
            labels.append("AI")
        elif s <= threshold_low:
            labels.append("HUMAN")
        else:
            labels.append("UNCERTAIN")

    transitions = 0
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1] and labels[idx] != "UNCERTAIN" and labels[idx - 1] != "UNCERTAIN":
            transitions += 1

    # Mixed authorship condition: significant presence of both classes with high variance
    has_both_strong = (ai_count >= 1 and human_count >= 1)
    is_mixed = False
    mixed_confidence = 0.0

    if has_both_strong and var >= variance_threshold:
        is_mixed = True
        mixed_confidence = min(0.95, float(0.5 + std + (transitions * 0.1)))
        assessment = (
            f"Detected mixed authorship: {ai_count} chunk(s) strongly AI, "
            f"{human_count} chunk(s) strongly human (score variance: {var:.4f})."
        )
    elif var >= (variance_threshold * 1.5):
        is_mixed = True
        mixed_confidence = min(0.85, float(0.4 + std))
        assessment = f"High stylistic variance across sections (score std: {std:.3f})."
    else:
        is_mixed = False
        assessment = "Document exhibits uniform stylistic consistency across sections."

    return {
        "is_mixed": is_mixed,
        "mixed_confidence": round(mixed_confidence, 3),
        "score_variance": round(var, 4),
        "score_std": round(std, 3),
        "ai_chunk_count": ai_count,
        "human_chunk_count": human_count,
        "total_chunks": n,
        "ai_ratio": round(ai_count / max(1, n), 3),
        "human_ratio": round(human_count / max(1, n), 3),
        "transitions": transitions,
        "assessment": assessment,
    }


def aggregate_chunk_predictions(
    chunk_results: List[Dict[str, Any]],
    decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Intelligently aggregate chunk-level predictions into a document-level forensic report.
    Uses robust statistics (trimmed mean, median, IQR) and peak suspicion tracking.
    """
    if not chunk_results:
        return {
            "overall_score": 0.5,
            "overall_label": "UNCERTAIN / LOW-CONFIDENCE",
            "decision_confidence": 0.0,
            "chunk_count": 0,
            "mixed_analysis": {},
            "suspicious_chunks": [],
        }

    scores = [c["prediction_score"] for c in chunk_results]
    arr = np.array(scores, dtype=float)

    mean_score = float(np.mean(arr))
    median_score = float(np.median(arr))
    max_score = float(np.max(arr))
    min_score = float(np.min(arr))
    p75 = float(np.percentile(arr, 75))
    p25 = float(np.percentile(arr, 25))
    iqr = p75 - p25

    # Mixed authorship check
    mixed_info = detect_mixed_authorship(scores)

    # Weighted composite score:
    # Emphasize median to resist single-chunk outliers, but consider top quartile for localized AI text
    if len(scores) >= 3:
        # 60% median, 25% 75th percentile (peak suspicion), 15% mean
        composite_score = float(0.60 * median_score + 0.25 * p75 + 0.15 * mean_score)
    else:
        composite_score = mean_score

    # Determine 4-tier classification
    if mixed_info["is_mixed"] and mixed_info["mixed_confidence"] >= 0.50:
        tier_label = "MIXED CHARACTERISTICS"
    elif composite_score >= 0.58:
        tier_label = "AI-GENERATED"
    elif composite_score <= 0.42:
        tier_label = "HUMAN-WRITTEN"
    else:
        tier_label = "UNCERTAIN / LOW-CONFIDENCE"

    # Confidence calculation (distance from ambiguity center 0.5, modulated by variance)
    raw_strength = abs(composite_score - 0.5) * 2.0
    # Penalty if chunk variance is high or mixed
    confidence_penalty = mixed_info["score_std"] * 0.4
    calibrated_confidence = max(0.1, min(0.99, raw_strength - confidence_penalty))

    # Identify suspicious chunks (AI-leaning)
    suspicious = [
        c for c in chunk_results if c["prediction_score"] >= 0.55
    ]
    suspicious_sorted = sorted(suspicious, key=lambda x: x["prediction_score"], reverse=True)

    return {
        "composite_score": round(composite_score, 4),
        "mean_score": round(mean_score, 4),
        "median_score": round(median_score, 4),
        "max_score": round(max_score, 4),
        "min_score": round(min_score, 4),
        "iqr": round(iqr, 4),
        "tier_label": tier_label,
        "calibrated_confidence": round(calibrated_confidence, 3),
        "confidence_percentage": round(composite_score * 100.0, 1),
        "chunk_count": len(chunk_results),
        "mixed_analysis": mixed_info,
        "suspicious_chunks": suspicious_sorted,
    }

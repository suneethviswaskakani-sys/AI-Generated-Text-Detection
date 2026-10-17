"""
Structural / stylometric feature extraction.

No target labels are used in feature construction (leakage-safe by design).
If `textstat` is unavailable, readability/syllable features fall back to a
lightweight built-in approximation instead of failing.
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

try:
    from textstat import flesch_reading_ease as _flesch_ext, syllable_count as _syllables_ext
    _HAVE_TEXTSTAT = True
except Exception:
    _HAVE_TEXTSTAT = False

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
PARA_RE = re.compile(r"\n\s*\n+")
VOWEL_RE = re.compile(r"[aeiouyAEIOUY]+")

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "by", "from", "at", "is", "are", "was", "were", "be",
    "been", "this", "that", "it", "we", "you", "they", "he", "she", "not",
    "so", "than", "too", "very", "can", "will", "would", "could", "should",
}


def _entropy(items):
    if not items:
        return 0.0
    c = Counter(items)
    n = len(items)
    return float(-sum((v / n) * math.log2(v / n) for v in c.values()))


def _syllables_in_word(word: str) -> int:
    groups = VOWEL_RE.findall(word)
    count = len(groups)
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _readability_and_syllables(text: str, words, sentences):
    """Use textstat if available; otherwise a documented Flesch approximation."""
    if _HAVE_TEXTSTAT:
        try:
            return float(_flesch_ext(text)), int(_syllables_ext(text))
        except Exception:
            pass
    syllables = sum(_syllables_in_word(w) for w in words) if words else 0
    n_words = max(1, len(words))
    n_sent = max(1, len(sentences))
    # Standard Flesch Reading Ease formula, computed manually as a fallback.
    fre = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)
    return float(fre), int(syllables)


def extract_structural_features(text: str) -> dict:
    text = str(text or "")
    words = WORD_RE.findall(text)
    sentences = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    paragraphs = [p.strip() for p in PARA_RE.split(text) if p.strip()]

    lower_words = [w.lower() for w in words]
    word_lengths = [len(w) for w in words]
    sent_lengths = [len(WORD_RE.findall(s)) for s in sentences]
    para_lengths = [len(WORD_RE.findall(p)) for p in paragraphs]

    word_counts = Counter(lower_words)
    unique_words = len(word_counts)
    hapax = sum(1 for c in word_counts.values() if c == 1)
    n_words = len(words)
    n_chars = len(text)
    n_sent = max(1, len(sentences))

    punct = sum(ch in ".,;:!?-'\"()[]{}" for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    upper = sum(ch.isupper() for ch in text)
    digits = sum(ch.isdigit() for ch in text)

    fw = sum(w in FUNCTION_WORDS for w in lower_words)
    stopword_ratio = fw / max(1, n_words)

    bigrams = list(zip(lower_words, lower_words[1:]))
    repeated_bigrams = len(bigrams) - len(set(bigrams)) if bigrams else 0

    word_entropy = _entropy(lower_words)
    sent_entropy = _entropy(sent_lengths)

    avg_sent = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_std = float(np.std(sent_lengths)) if sent_lengths else 0.0
    avg_word = float(np.mean(word_lengths)) if word_lengths else 0.0
    word_std = float(np.std(word_lengths)) if word_lengths else 0.0
    avg_para = float(np.mean(para_lengths)) if para_lengths else 0.0

    repetition = 1.0 - (unique_words / n_words) if n_words else 0.0
    readability, syllables = _readability_and_syllables(text, words, sentences)

    return {
        "char_count": n_chars,
        "word_count": n_words,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_word_length": avg_word,
        "word_length_std": word_std,
        "avg_sentence_length": avg_sent,
        "sentence_length_std": sent_std,
        "avg_paragraph_length": avg_para,
        "lexical_diversity": unique_words / max(1, n_words),
        "type_token_ratio": unique_words / max(1, n_words),
        "hapax_ratio": hapax / max(1, unique_words),
        "repetition_ratio": repetition,
        "bigram_repetition": repeated_bigrams / max(1, len(bigrams)),
        "punctuation_density": punct / max(1, n_chars),
        "comma_ratio": text.count(",") / max(1, n_chars),
        "question_ratio": text.count("?") / n_sent,
        "exclamation_ratio": text.count("!") / n_sent,
        "uppercase_ratio": upper / max(1, alpha),
        "digit_ratio": digits / max(1, n_chars),
        "function_word_ratio": stopword_ratio,
        "stopword_ratio": stopword_ratio,
        "word_entropy": word_entropy,
        "sentence_entropy": sent_entropy,
        "burstiness": sent_std / max(1.0, avg_sent),
        "readability_flesch": readability,
        "syllables_per_word": syllables / max(1, n_words),
    }


def vectorize_structural(texts) -> pd.DataFrame:
    rows = [extract_structural_features(t) for t in texts]
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0)
    return df


FEATURE_NAMES = list(extract_structural_features("Sample text for column names.").keys())


AI_DISCOURSE_MARKERS = {
    "furthermore", "moreover", "in conclusion", "additionally", "consequently",
    "it is important to note", "it is crucial", "plays a crucial role",
    "plays a pivotal role", "delve into", "a testament to", "in summary",
    "on the other hand", "first and foremost", "it is worth noting",
    "by exploring", "in today's world", "in the realm of", "not only",
    "seamlessly", "harnessing the power", "fostering", "paramount"
}


def detect_suspicious_patterns(text: str) -> dict:
    """
    Forensic analysis of suspicious stylistic patterns typical of AI-generated text:
    - Over-reliance on formal discourse/transition markers
    - Abnormally low sentence length burstiness (metronomic cadence)
    - Monotonous sentence openers
    - Flat vocabulary distribution
    """
    text = str(text or "")
    words = WORD_RE.findall(text.lower())
    sentences = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    n_words = len(words)
    n_sent = max(1, len(sentences))

    # 1. Discourse markers count
    lower_text = text.lower()
    found_markers = []
    for marker in AI_DISCOURSE_MARKERS:
        cnt = lower_text.count(marker)
        if cnt > 0:
            found_markers.append((marker, cnt))

    marker_instances = sum(cnt for _, cnt in found_markers)
    marker_density = (marker_instances / max(1, n_words)) * 1000  # per 1000 words

    # 2. Burstiness check
    sent_lengths = [len(WORD_RE.findall(s)) for s in sentences]
    avg_sent = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_std = float(np.std(sent_lengths)) if sent_lengths else 0.0
    burstiness = sent_std / max(1.0, avg_sent)

    # 3. Sentence openers uniformity
    openers = []
    for s in sentences:
        s_words = WORD_RE.findall(s.lower())
        if s_words:
            openers.append(s_words[0])
    
    opener_counts = Counter(openers)
    top_opener_ratio = (max(opener_counts.values()) / n_sent) if opener_counts else 0.0

    # 4. Pattern flags & score
    flags = []
    suspicion_points = 0

    if marker_density >= 12.0:
        flags.append(f"High density of AI discourse markers ({marker_density:.1f} per 1000 words).")
        suspicion_points += 30
    elif marker_density >= 6.0:
        flags.append(f"Elevated discourse markers ({marker_density:.1f} per 1000 words).")
        suspicion_points += 15

    if burstiness < 0.40 and n_sent >= 4:
        flags.append(f"Extremely low sentence length variance (burstiness={burstiness:.2f}), indicative of synthetic cadence.")
        suspicion_points += 30
    elif burstiness < 0.55 and n_sent >= 4:
        flags.append(f"Low burstiness ({burstiness:.2f}) indicates uniform sentence structure.")
        suspicion_points += 15

    if top_opener_ratio >= 0.40 and n_sent >= 5:
        top_opener = opener_counts.most_common(1)[0][0]
        flags.append(f"Repetitive sentence starter '{top_opener}' ({top_opener_ratio*100:.0f}% of sentences).")
        suspicion_points += 20

    if not flags:
        flags.append("Natural stylistic variability observed; no anomalous synthetic patterns detected.")

    suspicion_score = min(100, suspicion_points + int(marker_density * 2.5))

    return {
        "discourse_marker_density": round(marker_density, 2),
        "found_discourse_markers": sorted(found_markers, key=lambda x: x[1], reverse=True)[:6],
        "burstiness": round(burstiness, 3),
        "burstiness_assessment": "Low / Synthetic" if burstiness < 0.50 else "Natural / Human-like",
        "sentence_length_mean": round(avg_sent, 1),
        "sentence_length_std": round(sent_std, 1),
        "top_opener_repetition": round(top_opener_ratio, 3),
        "flags": flags,
        "suspicion_index": suspicion_score,
    }

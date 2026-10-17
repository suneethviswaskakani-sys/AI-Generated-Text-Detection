"""
Robustness testing: how much does the ensemble's prediction change under
simple, non-semantic-changing perturbations? Per spec: "Do not claim that
the detector can reliably detect all AI-generated text. Report limitations
honestly." These are basic, dependency-free perturbations (no paraphrasing
API/model available in this environment) -- a genuine but limited robustness
check, not a full adversarial-attack evaluation.
"""
import random
import re

import numpy as np

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def perturb_lowercase(text: str) -> str:
    return text.lower()


def perturb_remove_punctuation(text: str) -> str:
    return re.sub(r"[.,;:!?]", "", text)


def perturb_shuffle_sentences(text: str, seed: int = 0) -> str:
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    if len(sentences) < 2:
        return text
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def perturb_extra_whitespace(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = text.split()
    return (" " * rng.randint(1, 3)).join(words)


def perturb_double_spacing_and_typos(text: str, seed: int = 0) -> str:
    """Randomly duplicate a handful of characters -- crude typo simulation."""
    rng = random.Random(seed)
    chars = list(text)
    n_typos = max(1, len(chars) // 200)
    for _ in range(n_typos):
        if len(chars) < 2:
            break
        i = rng.randint(0, len(chars) - 2)
        chars.insert(i, chars[i])
    return "".join(chars)


PERTURBATIONS = {
    "lowercase": perturb_lowercase,
    "remove_punctuation": perturb_remove_punctuation,
    "shuffle_sentences": perturb_shuffle_sentences,
    "extra_whitespace": perturb_extra_whitespace,
    "typo_simulation": perturb_double_spacing_and_typos,
}


def run_robustness_suite(predictor, texts, labels) -> dict:
    """For each perturbation, measure how often the ensemble's binary
    prediction flips relative to its prediction on the original text, and
    whether accuracy against ground truth drops."""
    results = {}
    originals = [predictor.predict(t) for t in texts]
    orig_labels = [1 if r["label"] == "AI-GENERATED" else 0 for r in originals]
    orig_correct = np.mean([int(o == l) for o, l in zip(orig_labels, labels)])

    for name, fn in PERTURBATIONS.items():
        flips = 0
        perturbed_correct = 0
        for text, label, orig_pred in zip(texts, labels, orig_labels):
            pt = fn(text)
            r = predictor.predict(pt)
            new_pred = 1 if r["label"] == "AI-GENERATED" else (0 if r["label"] == "HUMAN-WRITTEN" else orig_pred)
            if new_pred != orig_pred:
                flips += 1
            if new_pred == label:
                perturbed_correct += 1
        results[name] = {
            "flip_rate": flips / len(texts),
            "accuracy_after_perturbation": perturbed_correct / len(texts),
        }

    return {
        "n_examples": len(texts),
        "original_accuracy": float(orig_correct),
        "perturbations": results,
        "note": (
            "flip_rate = fraction of predictions that changed class after the perturbation "
            "(instability, not necessarily wrong). accuracy_after_perturbation compares the "
            "perturbed prediction to ground truth. These are basic textual perturbations, not "
            "semantic paraphrasing -- a model robust to these can still be fooled by a real "
            "paraphrase; no paraphrasing tool was available in this environment to test that."
        ),
    }

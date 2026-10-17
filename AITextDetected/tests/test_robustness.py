import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.robustness import (
    perturb_lowercase, perturb_remove_punctuation, perturb_shuffle_sentences,
    perturb_extra_whitespace, perturb_double_spacing_and_typos,
)
from src.error_analysis import analyze_errors
from src.calibration_eval import calibration_report


class TestLegacyRobustness(unittest.TestCase):
    def test_perturb_lowercase(self):
        self.assertEqual(perturb_lowercase("Hello WORLD"), "hello world")

    def test_perturb_remove_punctuation(self):
        self.assertEqual(perturb_remove_punctuation("Hello, world!"), "Hello world")

    def test_perturb_shuffle_sentences_preserves_content(self):
        text = "First sentence. Second sentence. Third sentence."
        out = perturb_shuffle_sentences(text, seed=1)
        self.assertIn("First sentence.", out)
        self.assertIn("Second sentence.", out)
        self.assertIn("Third sentence.", out)

    def test_perturb_shuffle_single_sentence_unchanged(self):
        text = "Only one sentence here"
        self.assertEqual(perturb_shuffle_sentences(text), text)

    def test_perturb_extra_whitespace(self):
        out = perturb_extra_whitespace("a b c", seed=0)
        self.assertEqual(out.replace(" ", ""), "abc")

    def test_perturb_typo_simulation_same_or_longer(self):
        text = "some example text here"
        out = perturb_double_spacing_and_typos(text, seed=0)
        self.assertGreaterEqual(len(out), len(text))

    def test_analyze_errors_counts(self):
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d"],
            "source": ["s1", "s2", "s3", "s4"],
            "prompt_name": ["p1", "p2", "p3", "p4"],
        })
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])  # index0=FP, index2=FN
        p_final = np.array([0.6, 0.2, 0.3, 0.9])
        report = analyze_errors(df, y_true, y_pred, p_final)
        self.assertEqual(report["n_false_positives"], 1)
        self.assertEqual(report["n_false_negatives"], 1)

    def test_calibration_report_shape(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = calibration_report(y, p, n_bins=5)
        self.assertIn("brier_score", report)
        self.assertLessEqual(len(report["bins"]), 5)


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import extract_structural_features, vectorize_structural, FEATURE_NAMES
from src.reliability import assess_reliability
from src.utils import normalize_label


class TestLegacyFeatures(unittest.TestCase):
    def test_empty_text(self):
        feats = extract_structural_features("")
        self.assertEqual(feats["word_count"], 0)
        self.assertEqual(feats["char_count"], 0)

    def test_normal_text(self):
        feats = extract_structural_features("This is a normal sentence. It has two sentences.")
        self.assertGreater(feats["word_count"], 0)
        self.assertEqual(feats["sentence_count"], 2)

    def test_special_characters(self):
        feats = extract_structural_features("!!! @@@ ### $$$ ??? ...")
        self.assertGreater(feats["punctuation_density"], 0)

    def test_unicode_text(self):
        feats = extract_structural_features("héllo wörld café naïve 日本語のテキスト")
        self.assertGreater(feats["word_count"], 0)

    def test_numbers_text(self):
        feats = extract_structural_features("There are 123 items and 456 more, totaling 579.")
        self.assertGreater(feats["digit_ratio"], 0)

    def test_vectorize_shape_and_no_nan(self):
        texts = ["Short text.", "A much longer piece of text with several sentences. " * 5, ""]
        df = vectorize_structural(texts)
        self.assertEqual(df.shape[0], 3)
        self.assertEqual(df.shape[1], len(FEATURE_NAMES))
        self.assertFalse(df.isna().any().any())
        self.assertTrue(np.isfinite(df.values).all())

    def test_reliability_tiers(self):
        self.assertEqual(assess_reliability(10)["tier"], "insufficient")
        self.assertFalse(assess_reliability(10)["usable"])
        self.assertEqual(assess_reliability(75)["factors"]["length"], "low")
        self.assertEqual(assess_reliability(150)["factors"]["length"], "moderate")
        self.assertEqual(assess_reliability(500)["factors"]["length"], "optimal")

    def test_normalize_label(self):
        self.assertEqual(normalize_label("AI"), 1)
        self.assertEqual(normalize_label(1), 1)
        self.assertEqual(normalize_label("Human"), 0)
        self.assertEqual(normalize_label(0), 0)
        self.assertEqual(normalize_label("AI-generated"), 1)


if __name__ == "__main__":
    unittest.main()

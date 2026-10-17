import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ensemble import equal_weights, auc_weights, f1_weights, combine, best_threshold


class TestLegacyEnsemble(unittest.TestCase):
    def test_equal_weights_sum_to_one(self):
        w = equal_weights(3)
        self.assertTrue(np.isclose(w.sum(), 1.0))
        self.assertTrue(np.allclose(w, [1 / 3, 1 / 3, 1 / 3]))

    def test_auc_weights_favor_better_model(self):
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        good = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95, 0.15, 0.85, 0.05, 0.9])
        bad = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        w = auc_weights(y, [good, bad])
        self.assertGreater(w[0], w[1])
        self.assertTrue(np.isclose(w.sum(), 1.0))

    def test_combine_matches_weighted_sum(self):
        weights = np.array([0.5, 0.5])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        combined = combine(weights, [p1, p2])
        self.assertTrue(np.allclose(combined, [0.5, 0.5]))

    def test_best_threshold_in_range(self):
        y = np.array([0] * 50 + [1] * 50)
        p = np.concatenate([
            np.random.RandomState(0).uniform(0, 0.4, 50),
            np.random.RandomState(1).uniform(0.6, 1.0, 50)
        ])
        t = best_threshold(y, p, (0.3, 0.7, 41))
        self.assertTrue(0.3 <= t <= 0.7)


if __name__ == "__main__":
    unittest.main()

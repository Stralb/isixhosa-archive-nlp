"""Fast deterministic tests. Run: python -m unittest discover -s factory/tests -v"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "independent"))
import run_experiment as exp


class FactoryTests(unittest.TestCase):
    def test_factorial(self):
        a = {"PP": 1.0, "PW": 2.0, "WP": 3.0, "WW": 4.0}
        self.assertEqual(exp.contrast(a), {
            "allocation": 2.0, "within_view": 1.0,
            "interaction": 0.0, "WW_minus_PP": 3.0,
        })

    def test_all_four_distributions_normalize(self):
        units = [
            dict(filename="a", view="en", nwords=20),
            dict(filename="a", view="xh", nwords=40),
            dict(filename="b", view="en", nwords=80),
            dict(filename="b", view="xh", nwords=20),
        ]
        for arm in exp.ARMS:
            indices, weights, p_en = exp.distribution(units, arm)
            self.assertEqual(len(indices["en"]), 2)
            self.assertEqual(len(indices["xh"]), 2)
            self.assertAlmostEqual(float(weights["en"].sum()), 1.0)
            self.assertAlmostEqual(float(weights["xh"].sum()), 1.0)
            self.assertTrue(0 < p_en < 1)

    def test_sample_deterministic(self):
        units = []
        for fname in ("a", "b"):
            for view in exp.VIEWS:
                units.append(dict(filename=fname, view=view, nwords=150,
                                  words=(["isi"] * 75 + ["xhosa"] * 75)))
        for arm in exp.ARMS:
            one, audit1 = exp.draw_sample(units, arm, 441, budget=2000)
            two, audit2 = exp.draw_sample(units, arm, 441, budget=2000)
            self.assertEqual(one, two)
            self.assertEqual(audit1, audit2)
            self.assertTrue(1970 <= audit1["characters"] <= 2000)

    def test_bigram_finite(self):
        class FakeTokenizer:
            def vocab_size(self):
                return 8

            def encode(self, text, out_type=int):
                return [1 if w == "a" else 2 for w in text.split()]

        model = exp.SmoothedBigram(FakeTokenizer(), [{"text": "a a b b"}])
        for alpha in exp.ALPHAS:
            self.assertTrue(np.isfinite(model.bits([1, 2, 1], alpha)))
            self.assertGreater(model.bits([1, 2, 1], alpha), 0)

    def test_heldout_window_gate(self):
        rows = [
            dict(filename="x", issue="IMVO_1888-01-01", year=1888, view="en", words=["a"] * 39),
            dict(filename="y", issue="IMVO_1888-01-08", year=1888, view="xh", words=["b"] * 100),
        ]
        result = exp.test_examples(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chars"], 100)
        self.assertEqual(result[0]["period"], "1888")


if __name__ == "__main__":
    unittest.main()

import itertools
import unittest
import numpy as np
from cka_prune import linear_cka, select


class CKATests(unittest.TestCase):
    def test_invariance_and_gram_feature_agreement(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(30, 7))
        q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        self.assertAlmostEqual(linear_cka(x, 3 * x @ q + 5), 1.)
        y = rng.normal(size=(30, 9))
        a, b = x - x.mean(0), y - y.mean(0)
        expected = np.sum((a @ a.T) * (b @ b.T)) / (np.linalg.norm(a @ a.T) * np.linalg.norm(b @ b.T))
        self.assertAlmostEqual(linear_cka(x, y), expected)
        self.assertAlmostEqual(linear_cka(np.tile(x, (1, 8)), np.tile(y, (1, 8))), expected)

    def test_reject_bad_activations(self):
        for x in [np.ones((5, 3)), np.full((5, 3), np.nan), np.ones((2, 3))]:
            with self.assertRaises(ValueError):
                linear_cka(x, x)

    def test_constrained_optimum_against_exhaustive_search(self):
        scores = [0., .98, .99, .97, .96, .99, .98, .95]
        for budget in [2, 4]:
            chosen = select(scores, budget, .95, 2)
            feasible = []
            for drop in itertools.combinations(range(1, len(scores)), budget):
                keep = [i for i in range(len(scores)) if i not in drop]
                if all(old % 2 == new % 2 for new, old in enumerate(keep)):
                    feasible.append(sum(scores[i] for i in drop))
            self.assertAlmostEqual(sum(scores[i] for i in chosen), max(feasible))

    def test_anchors_and_infeasible_budget(self):
        self.assertEqual(select([0, .99, .2, .98], 2, .95), [1, 3])
        with self.assertRaises(ValueError):
            select([0, .99, .2, .98], 3, .95)
        with self.assertRaises(ValueError):
            select([0, .99, .2, .1], 1, .95, 2)


if __name__ == '__main__':
    unittest.main()

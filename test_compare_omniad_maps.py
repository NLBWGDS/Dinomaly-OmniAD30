import unittest

import numpy as np

from compare_omniad_maps import aupro_hist, f1_hist, image_f1, region_hist


class MetricsTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        self.assertEqual(f1_hist(np.array([0, 10]), np.array([10, 0]))[0], 1)
        self.assertAlmostEqual(aupro_hist(np.array([0., 1.]), 1, np.array([10, 0])), 1)
        self.assertEqual(aupro_hist(np.array([1., 0.]), 1, np.array([0, 10])), 0)

    def test_constant_score_interpolation(self):
        self.assertAlmostEqual(aupro_hist(np.array([1.]), 1, np.array([100])), .15)
        self.assertIsNone(image_f1([.1, .2], [True, True]))
        self.assertEqual(image_f1([.1, .9], [False, True]), 1)

    def test_regions_equal_weight(self):
        truth = np.zeros((10, 10), dtype=bool)
        truth[0, 0] = True
        truth[4:7, 4:7] = True
        scores = np.zeros((10, 10))
        scores[0, 0] = .9
        scores[4:7, 4:7] = .1
        hist, regions = region_hist(scores, truth, np.array([0, .5, 1]))
        self.assertEqual(regions, 2)
        np.testing.assert_allclose(hist, [1, 1])


if __name__ == '__main__':
    unittest.main()

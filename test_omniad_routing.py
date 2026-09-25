import unittest

import numpy as np

from export_omniad_routed import fuse, proxy


class RoutingTests(unittest.TestCase):
    def test_independent_image_score(self):
        a = np.array([[0., .1], [.2, .9]])
        b = np.array([[.8, .7], [.6, .5]])
        for weight in (0, .5, 1):
            result, score = fuse(a, b, weight, .25)
            self.assertEqual(score, .9)
            np.testing.assert_allclose(result, (1-weight)*a + weight*b)

    def test_validation(self):
        with self.assertRaises(ValueError):
            fuse(np.ones((2, 2)), np.ones((3, 3)), .5, .01)
        with self.assertRaises(ValueError):
            fuse(np.ones((2, 2)), np.full((2, 2), np.nan), .5, .01)

    def test_weight_order(self):
        self.assertEqual(proxy(dict(pixel_f1=1, pixel_aupro=0, image_f1=0)), 26)
        self.assertEqual(proxy(dict(pixel_f1=0, pixel_aupro=1, image_f1=0)), 6)
        self.assertEqual(proxy(dict(pixel_f1=0, pixel_aupro=0, image_f1=1)), 18)


if __name__ == '__main__':
    unittest.main()

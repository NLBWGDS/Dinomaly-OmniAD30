import unittest

import numpy as np

from predict_omniad_tiled import MapStitcher, tile_boxes


class TilingTests(unittest.TestCase):
    def test_coordinate_reconstruction(self):
        for width, height in [(1481, 1294), (1058, 581), (19, 11), (1, 1)]:
            expected = np.arange(width * height, dtype=np.float32).reshape(height, width)
            stitch = MapStitcher(width, height)
            for box in tile_boxes(width, height, 2, .25):
                x0, y0, x1, y1 = box
                stitch.add(box, expected[y0:y1, x0:x1])
            np.testing.assert_allclose(stitch.finish(), expected, rtol=1e-6, atol=1e-5)

    def test_constant_map_has_no_seams(self):
        stitch = MapStitcher(137, 91)
        for x0, y0, x1, y1 in tile_boxes(137, 91, 3, .4):
            stitch.add((x0, y0, x1, y1), np.ones((y1-y0, x1-x0), np.float32))
        np.testing.assert_allclose(stitch.finish(), 1)

    def test_invalid_and_uncovered(self):
        with self.assertRaises(ValueError):
            tile_boxes(100, 100, 2, 1)
        with self.assertRaises(ValueError):
            MapStitcher(10, 10).finish()


if __name__ == '__main__':
    unittest.main()

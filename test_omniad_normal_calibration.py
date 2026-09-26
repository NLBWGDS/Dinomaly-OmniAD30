import unittest

import numpy as np

from omniad_normal_calibration import fit_normal_calibration, calibrate_border, blend_detail_map
from predict_omniad_fullcoverage import coverage_canvas
import torch


class NormalCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.box = (4, 4, 12, 12)
        self.maps = []
        for shift in np.linspace(-.01, .01, 21):
            normal = np.full((16, 16), .10 + shift, dtype=np.float32)
            normal[4:12, 4:12] = .04 + shift
            self.maps.append(normal)
        self.calibration = fit_normal_calibration(iter(self.maps), 16, self.box, grid=16)

    def test_removes_known_normal_bias_without_changing_center(self):
        query = self.maps[10].copy()
        query[0, 0] += .05
        corrected = calibrate_border(query, self.calibration, strength=1)
        np.testing.assert_array_equal(corrected[4:12, 4:12], query[4:12, 4:12])
        self.assertAlmostEqual(float(corrected[0, 1]), .04, places=5)
        self.assertAlmostEqual(float(corrected[0, 0] - corrected[0, 1]), .05, places=5)
        self.assertEqual(self.calibration['normal_images'], 21)

    def test_zero_strength_is_exact_control(self):
        np.testing.assert_array_equal(calibrate_border(self.maps[0], self.calibration, 0), self.maps[0])
        half = calibrate_border(self.maps[10], self.calibration, .5)
        self.assertAlmostEqual(float(half[0, 0]), .07, places=5)

    def test_low_resolution_and_bounded_gain(self):
        calibration = fit_normal_calibration(self.maps, 16, self.box, grid=4)
        calibration['high'] = calibration['median'].copy()
        query = np.full((16, 16), .5, dtype=np.float32)
        corrected = calibrate_border(query, calibration, 1)
        self.assertTrue(np.isfinite(corrected).all())
        self.assertTrue((corrected >= 0).all())
        self.assertLessEqual(float(corrected.max()), 4*.5 + .05)
        np.testing.assert_array_equal(corrected[4:12, 4:12], query[4:12, 4:12])

    def test_invalid_and_degenerate_fitting(self):
        with self.assertRaisesRegex(ValueError, 'At least five'):
            fit_normal_calibration(self.maps[:4], 16, self.box, 8)
        with self.assertRaises(ValueError):
            fit_normal_calibration([np.ones((16, 16))]*5, 16, self.box, 8)
        with self.assertRaises(ValueError):
            fit_normal_calibration([np.full((16, 16), np.nan)]*5, 16, self.box, 8)
        with self.assertRaises(ValueError):
            calibrate_border(self.maps[0], self.calibration, float('nan'))

    def test_coverage_to_calibration_pipeline(self):
        def canvas(scores):
            tensor = torch.from_numpy(scores[None])
            return coverage_canvas(tensor, 8, 2, lambda batch: batch[:, 0].numpy())
        calibration = fit_normal_calibration((canvas(x) for x in self.maps), 16, self.box, 8)
        raw = canvas(self.maps[10])
        corrected = calibrate_border(raw, calibration)
        self.assertEqual(corrected.shape, raw.shape)
        self.assertLess(float(corrected[0, 0]), float(raw[0, 0]))
        np.testing.assert_array_equal(corrected[4:12, 4:12], raw[4:12, 4:12])

    def test_detail_calibrates_center_and_matches_reference_scale(self):
        calibration = dict(self.calibration, reference_median=.02, reference_high=.029)
        query = self.maps[10].copy()
        query[7, 7] += .05
        baseline = np.full((16, 16), .08, dtype=np.float32)
        result = blend_detail_map(baseline, query, calibration, .25)
        self.assertAlmostEqual(float(result[0, 1]), .065, places=5)
        self.assertAlmostEqual(float(result[7, 8]), .065, places=5)
        self.assertAlmostEqual(float(result[7, 7] - result[7, 8]), .0125, places=5)
        np.testing.assert_array_equal(baseline, np.full((16, 16), .08, dtype=np.float32))
        np.testing.assert_array_equal(blend_detail_map(baseline, query, calibration, 0), baseline)

    def test_detail_original_size_and_invalid_inputs(self):
        baseline = np.full((23, 37), .04, dtype=np.float32)
        result = blend_detail_map(baseline, self.maps[10], self.calibration, .25)
        self.assertEqual(result.shape, (23, 37))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        for weight in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                blend_detail_map(baseline, self.maps[10], self.calibration, weight)
        with self.assertRaises(ValueError):
            blend_detail_map(baseline * np.nan, self.maps[10], self.calibration, .25)


if __name__ == '__main__':
    unittest.main()

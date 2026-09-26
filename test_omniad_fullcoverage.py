import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from export_omniad_routed import read_index
from predict_omniad_fullcoverage import coverage_boxes, CoverageStitcher, coverage_canvas, export_predictions


class CoverageTests(unittest.TestCase):
    def test_full_coverage_and_center_geometry(self):
        for side, crop in [(448, 392), (896, 392), (31, 9), (16, 16), (15, 10)]:
            boxes = coverage_boxes(side, crop)
            coverage = np.zeros((side, side), dtype=bool)
            offset = round((side-crop)/2)
            self.assertEqual(boxes[0], (offset, offset, offset+crop, offset+crop))
            for x0, y0, x1, y1 in boxes:
                self.assertEqual((x1-x0, y1-y0), (crop, crop))
                coverage[y0:y1, x0:x1] = True
            self.assertTrue(coverage.all())
        self.assertEqual(len(coverage_boxes(448, 392)), 5)
        self.assertEqual(len(coverage_boxes(896, 392)), 9)
        with self.assertRaises(ValueError):
            coverage_boxes(10, 11)

    def test_center_is_preserved_edges_are_predicted(self):
        boxes = coverage_boxes(16, 12)
        stitch = CoverageStitcher(16, boxes[0])
        for i, box in enumerate(boxes):
            stitch.add(box, np.full((12, 12), .2 if i == 0 else .8, np.float32))
        output = stitch.finish()
        np.testing.assert_array_equal(output[2:14, 2:14], np.full((12, 12), .2, np.float32))
        self.assertAlmostEqual(float(output[0, 0]), .8, places=6)
        self.assertTrue(np.isfinite(output).all())
        with self.assertRaises(ValueError):
            CoverageStitcher(16, boxes[0]).finish()

    def test_batched_inference_preserves_coordinates(self):
        tensor = torch.arange(31*31, dtype=torch.float32).reshape(1, 31, 31)
        def infer(batch):
            return batch[:, 0].numpy()
        for size in (1, 2, 7):
            result = coverage_canvas(tensor, 9, size, infer)
            np.testing.assert_allclose(result, tensor[0].numpy(), atol=1e-3)
        with self.assertRaises(ValueError):
            coverage_canvas(tensor, 9, 2, lambda batch: np.zeros((1, 1, 1)))

    def test_detail_stitches_all_windows_without_center_override(self):
        boxes = coverage_boxes(16, 12)
        stitch = CoverageStitcher(16, boxes[0], preserve_center=False)
        for i, box in enumerate(boxes):
            stitch.add(box, np.full((12, 12), .2 if i == 0 else .8, np.float32))
        output = stitch.finish()
        self.assertGreater(float(output[8, 8]), .2)
        self.assertLessEqual(float(output.max()), .800001)
        tensor = torch.arange(31*31, dtype=torch.float32).reshape(1, 31, 31)
        result = coverage_canvas(tensor, 9, 2, lambda batch: batch[:, 0].numpy(), preserve_center=False)
        np.testing.assert_allclose(result, tensor[0].numpy(), atol=1e-3)

    def test_export_preserves_unselected_maps_and_image_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, expected = {}, {}
            for category in ('nameplate7', 'wafer2'):
                path = root / 'data' / category / 'test/good/a.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(np.zeros((7, 9), dtype=np.uint8)).save(path)
                old = root / (category + '.npy')
                np.save(old, np.zeros((7, 9), dtype=np.float32))
                key = (category, 'good/a.png')
                expected[key] = path
                records[key] = ({'score': '0.12345678912345678'}, old)
            called = []
            def predict(path):
                called.append(path)
                return np.ones((7, 9), dtype=np.float32)
            output = root / 'output'
            export_predictions(records, expected, {'nameplate7'}, output, predict)
            self.assertEqual(len(called), 1)
            rows = read_index(output)
            self.assertEqual(rows[('wafer2', 'good/a.png')][1], root / 'wafer2.npy')
            self.assertTrue(np.load(rows[('nameplate7', 'good/a.png')][1]).all())
            self.assertTrue(all(row['score'] == '0.12345678912345678' for row, _ in rows.values()))
            with self.assertRaises(FileExistsError):
                export_predictions(records, expected, {'nameplate7'}, output, predict)


if __name__ == '__main__':
    unittest.main()

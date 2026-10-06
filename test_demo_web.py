import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from demo_web import DashboardData


class DashboardDataTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parent / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        root = Path(self.temp.name)
        self.data = root / "data"
        self.predictions = root / "predictions"
        image_path = self.data / "sample_part" / "test" / "good" / "000.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (24, 16), (110, 130, 140)).save(image_path)
        map_path = self.predictions / "sample_part" / "good" / "000.png.npy"
        map_path.parent.mkdir(parents=True)
        np.save(map_path, np.linspace(0, 1, 24 * 16, dtype=np.float32).reshape(16, 24))
        with (self.predictions / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["category", "image_path", "score", "map_path"])
            writer.writeheader()
            writer.writerow({"category": "sample_part", "image_path": str(image_path),
                             "score": "0.125", "map_path": str(map_path)})
        self.metrics = root / "metrics.json"
        self.metrics.write_text(json.dumps({
            "scheme": "unified baseline",
            "categories": {"sample_part": {
                "pixel_f1": 0.5, "image_f1": 0.75, "pixel_aupro": 0.8}},
            "mean": {"pixel_f1": 0.5, "image_f1": 0.75, "pixel_aupro": 0.8},
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_state_and_samples(self):
        dashboard = DashboardData(self.predictions, self.data, self.metrics)
        state = dashboard.state()
        self.assertEqual(state["images"], 1)
        self.assertEqual(state["categories"][0]["metrics"]["pixel_f1"], 0.5)
        self.assertFalse(dashboard.samples("sample_part")[0]["is_anomaly"])

    def test_all_views_are_png(self):
        dashboard = DashboardData(self.predictions, self.data, self.metrics)
        for view in ("original", "mask", "heatmap", "overlay"):
            payload, content_type = dashboard.render("sample_part", 0, view)
            expected = b"\x89PNG\r\n\x1a\n" if view == "mask" else b"\xff\xd8\xff"
            self.assertTrue(payload.startswith(expected), view)
            self.assertEqual(content_type, "image/png" if view == "mask" else "image/jpeg")

    def test_preview_is_downscaled_without_changing_export(self):
        dashboard = DashboardData(self.predictions, self.data, self.metrics, display_max_side=12)
        rgb, scores, truth = dashboard.sample_arrays("sample_part", 0)
        self.assertEqual(rgb.shape[:2], (8, 12))
        self.assertEqual(scores.shape, (8, 12))
        self.assertEqual(truth.shape, (8, 12))
        original = np.load(self.predictions / "sample_part" / "good" / "000.png.npy")
        self.assertEqual(original.shape, (16, 24))


if __name__ == "__main__":
    unittest.main()

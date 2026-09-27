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
            payload = dashboard.render("sample_part", 0, view)
            self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"), view)


if __name__ == "__main__":
    unittest.main()

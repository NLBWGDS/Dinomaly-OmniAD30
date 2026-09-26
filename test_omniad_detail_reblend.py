import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from export_omniad_routed import read_index
from reblend_omniad_detail import gated_reblend, normal_reliability, run, validate_source


class DetailReblendTests(unittest.TestCase):
    def setUp(self):
        self.stats = dict(median=np.zeros((2, 2), np.float32),
                          high=np.array([[.01, .02], [.04, .08]], np.float32),
                          reference_median=.04, reference_high=.05, normal_images=20)

    def test_normal_variability_reduces_reliability(self):
        np.testing.assert_allclose(normal_reliability(self.stats), [[1, .5], [.25, .25]])
        np.testing.assert_array_equal(normal_reliability(self.stats, 1), np.ones((2, 2)))

    def test_directional_gating_and_bounds(self):
        base = np.array([[.4, .4]], np.float32)
        detail = np.array([[.6, .2]], np.float32)
        result = gated_reblend(base, detail, np.array([[1, .5]], np.float32), .5)
        np.testing.assert_allclose(result, [[.6, .35]])
        self.assertTrue((result >= np.minimum(base, detail)).all())
        self.assertTrue((result <= np.maximum(base, detail)).all())
        np.testing.assert_array_equal(gated_reblend(base, detail, np.zeros((2, 2))), base)
        np.testing.assert_array_equal(gated_reblend(base, detail, np.ones((2, 2)), 1), detail)
        np.testing.assert_array_equal(base, np.array([[.4, .4]], np.float32))

    def test_original_shape_and_nonfinite_validation(self):
        base = np.full((17, 23), .4, np.float32)
        self.assertEqual(gated_reblend(base, base/2, normal_reliability(self.stats)).shape, base.shape)
        for value in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                gated_reblend(base, base, np.ones((2, 2)), value)
        with self.assertRaises(ValueError):
            gated_reblend(base, base*np.nan, np.ones((2, 2)))
        with self.assertRaises(ValueError):
            normal_reliability(dict(self.stats, high=np.full((2, 2), np.nan)))
        with self.assertRaises(ValueError):
            normal_reliability(dict(self.stats, normal_images=4))

    def test_rejects_wrong_provenance(self):
        manifest = dict(arguments=dict(detail_scale=2, normal_calibration=True,
                        predictions='original', detail_weight=.25), labels_used=False,
                        selected_categories=['ceramic_wafer'])
        validate_source(manifest, 'original', {'ceramic_wafer'})
        with self.assertRaises(ValueError):
            validate_source(manifest, 'already_blended', {'ceramic_wafer'})
        with self.assertRaises(ValueError):
            validate_source(manifest, 'original', {'chip1'})
        with self.assertRaises(ValueError):
            validate_source(dict(manifest, labels_used=True), 'original', {'ceramic_wafer'})

    def test_full30_export_preserves_unselected_candidate_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, detail, data, output = [root / name for name in ('base', 'detail', 'data', 'output')]
            base.mkdir()
            detail.mkdir()
            categories = ['ceramic_wafer', 'spindle_top'] + [f'other{i}' for i in range(28)]
            rows = {base: [], detail: []}
            for category in categories:
                (data / category / 'train/good').mkdir(parents=True)
                image = data / category / 'test/good/a.png'
                image.parent.mkdir(parents=True)
                Image.fromarray(np.zeros((3, 5), np.uint8)).save(image)
                for directory, value in ((base, .4), (detail, .2)):
                    path = directory / category / 'good/a.png.npy'
                    path.parent.mkdir(parents=True)
                    np.save(path, np.full((3, 5), value, np.float32))
                    rows[directory].append([category, str(image), '0.12345678912345678', str(path)])
            for directory in (base, detail):
                with (directory / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['category', 'image_path', 'score', 'map_path'])
                    writer.writerows(rows[directory])
            (detail / 'coverage_manifest.json').write_text(json.dumps(dict(
                arguments=dict(detail_scale=2, normal_calibration=True, detail_weight=.25, predictions=str(base)),
                selected_categories=categories[:2], labels_used=False)), encoding='utf-8')
            for category in categories[:2]:
                np.savez(detail / f'{category}_detail_calibration.npz', **self.stats)
            args = SimpleNamespace(baseline=str(base), detail_predictions=str(detail), output_dir=str(output),
                                   data_path=str(data), categories=','.join(categories[:2]),
                                   reliability_floor=.25, down_weight=.5)
            run(args)
            exported, original = read_index(output), read_index(detail)
            self.assertEqual(len(exported), 30)
            for key, (row, path) in exported.items():
                self.assertEqual(row['score'], original[key][0]['score'])
                if key[0] in categories[:2]:
                    result = np.load(path)
                    self.assertTrue((result >= .3 - 1e-6).all())
                    self.assertTrue((result <= .4).all())
                else:
                    self.assertEqual(path.resolve(), original[key][1].resolve())
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == '__main__':
    unittest.main()

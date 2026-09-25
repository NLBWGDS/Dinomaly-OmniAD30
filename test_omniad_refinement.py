import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from export_omniad_routed import read_index
from refine_omniad_maps import export_refined, refine_map


class RefinementTests(unittest.TestCase):
    def test_thin_line_halo(self):
        truth = np.zeros((64, 64))
        truth[:, 30:33] = 1
        blurred = gaussian_filter(truth, 3)
        refined = refine_map(blurred, truth, radius=8, eps=0.001, strength=0.5)
        self.assertLess(np.mean((refined - truth)**2), np.mean((blurred - truth)**2))
        self.assertLess(refined[32, 27], blurred[32, 27])
        self.assertGreater(refined[32, 31] / refined[32, 27],
                           blurred[32, 31] / blurred[32, 27])

    def test_identity_and_constant(self):
        guide = np.random.default_rng(7).random((23, 19))
        np.testing.assert_allclose(refine_map(guide, guide, strength=0), guide, rtol=1e-6)
        np.testing.assert_array_equal(refine_map(np.full(guide.shape, .25), guide), .25)

    def test_invalid_inputs(self):
        g = np.zeros((4, 4))
        for kwargs in ({'radius': 0}, {'eps': 0}, {'eps': float('nan')}, {'strength': 2}):
            with self.assertRaises(ValueError):
                refine_map(g, g, **kwargs)
        for p, guide in ((g, g[:2]), (g, g + 2), (g + np.nan, g)):
            with self.assertRaises(ValueError):
                refine_map(p, guide)

    def test_export_preserves_scores_and_other_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, data, target = root / 'source', root / 'data', root / 'target'
            source.mkdir()
            with (source / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['category', 'image_path', 'score', 'map_path'])
                for category in ('wafer2', 'iron_lattice'):
                    image = data / category / 'test' / 'bad' / 'a.png'
                    image.parent.mkdir(parents=True)
                    Image.fromarray(np.full((12, 16), 128, dtype=np.uint8)).save(image)
                    local = source / category / 'bad' / 'a.png.npy'
                    local.parent.mkdir(parents=True)
                    np.save(local, np.arange(192, dtype=np.float32).reshape(12, 16))
                    # Simulate copied exports with stale absolute map paths.
                    writer.writerow([category, str(image), '0.12345678912345678',
                                     '/old/server/' + category + '/bad/a.png.npy'])
            export_refined(source, data, target)
            records = read_index(target)
            self.assertEqual(len(records), 2)
            for row, _ in records.values():
                self.assertEqual(row['score'], '0.12345678912345678')
            self.assertEqual((source / 'iron_lattice/bad/a.png.npy').read_bytes(),
                             (target / 'iron_lattice/bad/a.png.npy').read_bytes())
            self.assertFalse(np.array_equal(np.load(source / 'wafer2/bad/a.png.npy'),
                                           np.load(target / 'wafer2/bad/a.png.npy')))
            with self.assertRaises(FileExistsError):
                export_refined(source, data, target)


if __name__ == '__main__':
    unittest.main()

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from diagnose_omniad import counts, read_pair, run


class DiagnosticTests(unittest.TestCase):
    def test_counts(self):
        result = counts(np.array([[.9, .8], [.2, .1]]),
                        np.array([[True, False], [True, False]]), .5)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertEqual(result['f1'], .5)

    def test_export_and_portable_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, pred, output = root / 'data', root / 'pred', root / 'out'
            image = data / 'sample/test/defect/001.png'
            mask = data / 'sample/ground_truth/defect/001_mask.png'
            maps = pred / 'sample/defect__001.npy'
            for path in (image, mask, maps):
                path.parent.mkdir(parents=True, exist_ok=True)
            truth = np.zeros((20, 30), dtype=np.uint8)
            truth[5:10, 12:18] = 255
            Image.new('RGB', (30, 20), 'gray').save(image)
            Image.fromarray(truth).save(mask)
            np.save(maps, truth.astype(np.float32) / 255)
            with (pred / 'scores.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['category', 'image_path', 'map_path', 'score'])
                writer.writeheader()
                writer.writerow(dict(category='sample', image_path='/server/data/sample/test/defect/001.png',
                                     map_path='/server/pred/sample/defect__001.npy', score=.9))
            run(SimpleNamespace(predictions=pred, data_path=data, output_dir=output,
                                categories=None, bins=64, top_k=1))
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['categories']['sample']['pixel_f1'], 1)
            with Image.open(output / 'sample/fp_01.png') as im:
                self.assertEqual(im.width, 1760)
            np.save(maps, np.zeros((10, 10)))
            with self.assertRaises(ValueError):
                read_pair((image, maps, mask))


if __name__ == '__main__':
    unittest.main()

import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluate_omniad_export import evaluate


class ExportEvaluationTests(unittest.TestCase):
    def test_image_metric_reads_csv_not_map(self):
        content = ('category,image_path,map_path,score\n'
                   'sample,bad,bad.npy,0.1\n'
                   'sample,good,good.npy,0.9\n')
        truth = np.array([[True, False], [False, False]])

        def locate_row(row, predictions, data):
            return Path(row['image_path']), Path(row['map_path']), ('mask' if row['image_path'] == 'bad' else None)

        def pair(paths):
            if str(paths[0]) == 'bad':
                return truth.astype(float), truth
            return np.zeros((2, 2)), np.zeros((2, 2), dtype=bool)

        with patch('pathlib.Path.open', return_value=io.StringIO(content)), \
                patch('evaluate_omniad_export.locate', side_effect=locate_row), \
                patch('evaluate_omniad_export.read_pair', side_effect=pair):
            metrics = evaluate('predictions', 'data', 64)['categories']['sample']
        self.assertEqual(metrics['pixel_f1'], 1.)
        self.assertEqual(metrics['pixel_aupro'], 1.)
        # CSV intentionally reverses the ranking despite perfect anomaly maps.
        self.assertAlmostEqual(metrics['image_f1'], 2/3)


if __name__ == '__main__':
    unittest.main()

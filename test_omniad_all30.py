import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from evaluate_omniad_all30 import (dataset_inventory, require_exact_keys, combine_predictions,
                                  prediction_command, write_comparison)
from evaluate_omniad_export import evaluate
from export_omniad_routed import read_index


class AllCategoryTests(unittest.TestCase):
    def test_missing_extra_and_nonfinite(self):
        key = ('wafer2', 'good/a.png')
        with self.assertRaises(ValueError):
            require_exact_keys({}, {key: None}, 'test')
        with self.assertRaises(ValueError):
            require_exact_keys({key: ({'score': '1'}, Path('x'))}, {}, 'test')
        with self.assertRaises(ValueError):
            require_exact_keys({key: ({'score': 'nan'}, Path('x'))}, {key: None}, 'test')

    def test_prediction_command_covers_explicit_categories(self):
        args = SimpleNamespace(data_path='data', checkpoint='unified.pth', batch_size=2,
                               num_workers=4, device='cuda:0')
        command = prediction_command(args, ['a', 'b'], 'out')
        self.assertEqual(command[command.index('--categories') + 1], 'a,b')
        self.assertEqual(command[command.index('--mode') + 1], 'predict')

    def test_export_and_evaluate_same_full_image_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, base, override = root / 'data', root / 'base', root / 'override'
            truth = np.array([[1, 0], [0, 0]], dtype=np.uint8)
            for category in ('wafer2', 'other'):
                (data / category / 'train/good').mkdir(parents=True)
                for label in ('good', 'bad'):
                    path = data / category / 'test' / label / 'a.png'
                    path.parent.mkdir(parents=True)
                    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)
                gt = data / category / 'ground_truth/bad/a_mask.png'
                gt.parent.mkdir(parents=True)
                Image.fromarray(truth * 255).save(gt)
            categories, expected = dataset_inventory(data, expected_categories=2)
            self.assertEqual(categories, ['other', 'wafer2'])
            with self.assertRaises(ValueError):
                dataset_inventory(data)
            for folder, subset in [(base, expected), (override, {k: v for k, v in expected.items() if k[0] == 'wafer2'})]:
                folder.mkdir()
                with (folder / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['category', 'image_path', 'score', 'map_path'])
                    for (category, relative), image in subset.items():
                        target = folder / category / (relative + '.npy')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        is_bad = relative.startswith('bad/')
                        scores = truth.astype(np.float32) if folder == override and is_bad else np.zeros((2, 2), np.float32)
                        np.save(target, scores)
                        writer.writerow([category, str(image), '0.8' if is_bad else '0.2', str(target)])
            combined = root / 'combined'
            sources = combine_predictions(base, override, expected, combined, {'wafer2'})
            chosen = read_index(combined)
            for key, (row, path) in chosen.items():
                correct = override if key[0] == 'wafer2' else base
                self.assertEqual(path, correct / key[0] / (key[1] + '.npy'))
                self.assertEqual(row['score'], '0.8' if key[1].startswith('bad/') else '0.2')
            with self.assertRaises(FileExistsError):
                combine_predictions(base, override, expected, combined, {'wafer2'})
            with self.assertRaises(ValueError):
                combine_predictions(base, base, expected, root / 'wrong', {'wafer2'})
            a, b = evaluate(base, data, 64), evaluate(combined, data, 64)
            self.assertEqual(a['image_ids'], b['image_ids'])
            self.assertEqual(a['categories']['other'], b['categories']['other'])
            self.assertEqual(b['categories']['wafer2']['pixel_f1'], 1.)
            self.assertEqual(a['mean']['image_f1'], b['mean']['image_f1'])
            report = root / 'comparison.csv'
            write_comparison(a, b, sources, report)
            with report.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['category'] for r in rows], ['other', 'wafer2', 'Mean'])
            self.assertEqual(float(rows[0]['delta_pixel_f1']), 0.)
            self.assertGreater(float(rows[1]['delta_pixel_f1']), 0.)
            b['image_ids'] = []
            with self.assertRaises(ValueError):
                write_comparison(a, b, sources, root / 'bad.csv')


if __name__ == '__main__':
    unittest.main()

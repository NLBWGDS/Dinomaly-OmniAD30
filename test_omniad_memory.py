import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from omniad_memory import descriptors, valid_patches, nearest_distance, calibration_scale, coreset_indices
from predict_omniad_memory import run


class MemoryTests(unittest.TestCase):
    def test_coreset_coverage_and_reproducibility(self):
        x = torch.tensor([[0., 0.]] * 20 + [[10., 0.], [0., 10.]])
        selected = coreset_indices(x, 3, seed=7)
        torch.testing.assert_close(selected, coreset_indices(x, 3, seed=7))
        self.assertEqual(selected.unique().numel(), 3)
        self.assertEqual(torch.cdist(x, x[selected]).min(dim=1).values.max().item(), 0.)
        duplicate = coreset_indices(torch.ones(12, 5), 8, projection_dim=2)
        self.assertEqual(duplicate.unique().numel(), 8)
        torch.testing.assert_close(coreset_indices(x, len(x)), torch.arange(len(x)))
        with self.assertRaises(ValueError):
            coreset_indices(x, len(x) + 1)

    def test_weighted_cosine_and_chunking(self):
        features = [torch.tensor([[[[1., 0.]], [[0., 1.]]]]),
                    torch.tensor([[[[1., 1.]], [[0., 0.]]]])]
        x = descriptors(features, [.25, .75])
        torch.testing.assert_close(x.norm(dim=1), torch.ones(2))
        torch.testing.assert_close(nearest_distance(x, x[:1], chunk=1), torch.tensor([0., .25]))
        torch.testing.assert_close(nearest_distance(x, x, chunk=1), nearest_distance(x, x, chunk=100))

    def test_calibration_excludes_entire_image(self):
        bank = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        owners = torch.tensor([0, 0, 1])
        result = nearest_distance(bank[:1], bank, owners=owners, exclude_owner=0)
        torch.testing.assert_close(result, torch.ones(1))
        with self.assertRaises(ValueError):
            nearest_distance(bank[:1], bank, owners=torch.zeros(3), exclude_owner=0)

    def test_padding_and_scale(self):
        mask = valid_patches(28, 56, 4, 4, 56).reshape(4, 4)
        self.assertFalse(mask[0].any())
        self.assertTrue(mask[1:3].all())
        self.assertFalse(mask[3].any())
        torch.testing.assert_close(torch.tensor(calibration_scale(torch.arange(1., 10.),
                                                                 torch.arange(1., 10.) * 2)),
                                   torch.tensor(.5))
        with self.assertRaises(ValueError):
            calibration_scale(torch.ones(10), torch.zeros(10))

    def test_end_to_end_no_masks_and_preserved_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, data, output = root / 'source', root / 'data', root / 'output'
            source.mkdir()
            rng = np.random.default_rng(42)
            category = 'infusion_bottle_bottom5'
            for i in range(2):
                path = data / category / 'train/good' / f'{i}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(1, 255, (16, 16, 3), dtype=np.uint8)).save(path)
            with (source / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['category', 'image_path', 'score', 'map_path'])
                for name in (category, 'wafer2'):
                    image_path = data / name / 'test/defect/a.png'
                    image_path.parent.mkdir(parents=True)
                    Image.fromarray(rng.integers(1, 255, (16, 16, 3), dtype=np.uint8)).save(image_path)
                    path = source / name / 'defect/a.png.npy'
                    path.parent.mkdir(parents=True)
                    np.save(path, np.full((16, 16), .2, dtype=np.float32))
                    writer.writerow([name, str(image_path), '0.12345678912345678', str(path)])

            def model(batch):
                features = F.avg_pool2d(batch, 4)
                return [features, features], [features.roll(1, 1), features.roll(1, 1)]

            def preprocess(args, checkpoint):
                args.crop_size, args.preprocess = 16, 'letterbox'

            fake = SimpleNamespace(
                load_checkpoint_model=lambda args, device: (model, {}),
                apply_checkpoint_preprocessing=preprocess,
                get_omniad_transforms=lambda args: (lambda im: torch.tensor(np.array(im)).permute(2, 0, 1).float()/255, None),
                restore_anomaly_map=lambda x, h, w, args: F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False),
                setup_seed=lambda seed: torch.manual_seed(seed))
            args = SimpleNamespace(predictions=str(source), data_path=str(data), output_dir=str(output),
                                   checkpoint='unused.pth', category=category, feature_weights='0.25,0.75',
                                   memory_weight=.25, bank_size=16, query_chunk=4, seed=1, device='cpu')
            with patch.dict('sys.modules', {'dinomaly_omniad_uni': fake}):
                run(args)
                with self.assertRaises(FileExistsError):
                    run(args)
            with (output / 'scores.csv').open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['score'] == '0.12345678912345678' for r in rows))
            self.assertEqual((source / 'wafer2/defect/a.png.npy').read_bytes(),
                             (output / 'wafer2/defect/a.png.npy').read_bytes())
            result = np.load(output / category / 'defect/a.png.npy')
            self.assertEqual(result.shape, (16, 16))
            self.assertTrue(np.isfinite(result).all())
            self.assertFalse(np.allclose(result, .2))
            self.assertTrue((output / 'normal_bank.pt').is_file())
            args.output_dir = str(root / 'coreset')
            args.sampling, args.candidate_multiplier, args.projection_dim = 'coreset', 2, 2
            with patch.dict('sys.modules', {'dinomaly_omniad_uni': fake}):
                run(args)
            artifact = torch.load(root / 'coreset/normal_bank.pt', weights_only=True)
            self.assertEqual(artifact['sampling'], 'coreset')
            self.assertEqual(artifact['bank'].shape[0], 16)
            self.assertEqual(artifact['candidate_count'], 32)
            self.assertGreaterEqual(artifact['owners'].unique().numel(), 2)


if __name__ == '__main__':
    unittest.main()

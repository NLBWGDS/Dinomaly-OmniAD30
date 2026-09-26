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
from predict_omniad_memory import run, configure_memory_geometry, memory_valid_mask


class MemoryTests(unittest.TestCase):
    def test_distinct_image_neighbors_ignore_duplicate_patches(self):
        query = torch.tensor([[1., 0.], [0., 1.]])
        bank = torch.tensor([[1., 0.]]*5 + [[0., 1.], [-1., 0.]])
        owners = torch.tensor([10]*5 + [20, 30])
        old = nearest_distance(query, bank)
        torch.testing.assert_close(nearest_distance(query, bank, owners=owners, neighbor_images=1),
                                   old, rtol=0, atol=0)
        result = nearest_distance(query, bank, chunk=1, owners=owners, neighbor_images=3)
        torch.testing.assert_close(result, torch.tensor([1., 2/3]))
        torch.testing.assert_close(result, nearest_distance(query, bank, chunk=8, owners=owners, neighbor_images=3))
        torch.testing.assert_close(nearest_distance(query[:1], bank, owners=owners,
                                                   exclude_owner=10, neighbor_images=2), torch.tensor([1.5]))
        for kwargs in (dict(neighbor_images=2), dict(owners=owners, neighbor_images=4),
                       dict(owners=owners, exclude_owner=10, neighbor_images=3), dict(neighbor_images=0)):
            with self.assertRaises(ValueError):
                nearest_distance(query, bank, **kwargs)

    def test_legacy_geometry_is_explicit_and_has_no_padding(self):
        config = SimpleNamespace(preprocess='legacy', image_size=448, crop_size=392)
        with self.assertRaises(ValueError):
            configure_memory_geometry(config)
        self.assertEqual(config.image_size, 448)
        original = configure_memory_geometry(config, True)
        self.assertEqual(original['image_size'], 448)
        self.assertEqual(config.image_size, 392)
        self.assertTrue(memory_valid_mask(config, 100, 300, 28, 28).all())
        config = SimpleNamespace(preprocess='letterbox', image_size=56, crop_size=56)
        configure_memory_geometry(config)
        torch.testing.assert_close(memory_valid_mask(config, 28, 56, 4, 4), valid_patches(28, 56, 4, 4, 56))
        with self.assertRaises(ValueError):
            configure_memory_geometry(config, True)

    def test_context_preserves_detail_and_adds_neighbor_evidence(self):
        a = torch.zeros(1, 2, 3, 3)
        a[:, 1] = 1
        a[:, :, 1, 1] = torch.tensor([1., 0.])
        b = a.clone()
        b[:, 1] *= -1
        detail_a, detail_b = descriptors([a], [1.]), descriptors([b], [1.])
        mixed_a = descriptors([a], [1.], .25)
        mixed_b = descriptors([b], [1.], .25)
        torch.testing.assert_close(descriptors([a], [1.], 0), detail_a, rtol=0, atol=0)
        torch.testing.assert_close(detail_a[4], detail_b[4])
        torch.testing.assert_close(mixed_a[:, :2], detail_a * (.75 ** .5))
        torch.testing.assert_close(mixed_a.norm(dim=1), torch.ones(9))
        self.assertGreater((1 - mixed_a[4] @ mixed_b[4]).item(), .1)

    def test_context_ignores_padding_values(self):
        mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        mask[:, 1, 1] = True
        a = torch.ones(1, 2, 3, 3)
        b = a.clone()
        b[:, :, ~mask[0]] = 1000
        b[:, 1, ~mask[0]] = -1000
        x = descriptors([a], [1.], .25, valid_mask=mask)
        y = descriptors([b], [1.], .25, valid_mask=mask)
        torch.testing.assert_close(x[4], y[4])
        for weight, kernel in ((-1., 3), (float('nan'), 3), (.25, 2)):
            with self.assertRaises(ValueError):
                descriptors([a], [1.], weight, kernel)

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
                for name in (category, 'wafer2', 'iron_lattice'):
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
                get_omniad_transforms=lambda args: (lambda im: torch.tensor(np.array(im.resize((16, 16)))).permute(2, 0, 1).float()/255, None),
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
            self.assertEqual(len(rows), 3)
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
            args.output_dir = str(root / 'context')
            args.sampling, args.context_weight, args.context_kernel = 'random', .25, 3
            with patch.dict('sys.modules', {'dinomaly_omniad_uni': fake}):
                run(args)
            contextual = torch.load(root / 'context/normal_bank.pt', weights_only=True)
            original = torch.load(output / 'normal_bank.pt', weights_only=True)
            torch.testing.assert_close(contextual['owners'], original['owners'])
            torch.testing.assert_close(contextual['bank'][:, :6], original['bank'] * (.75 ** .5))
            self.assertEqual(contextual['bank'].shape, (16, 12))
            with (root / 'context/scores.csv').open(newline='', encoding='utf-8') as handle:
                context_rows = list(csv.DictReader(handle))
            self.assertEqual([r['score'] for r in context_rows], [r['score'] for r in rows])
            self.assertEqual((source / 'wafer2/defect/a.png.npy').read_bytes(),
                             (root / 'context/wafer2/defect/a.png.npy').read_bytes())
            # Apply another category to the accepted bottle export, without
            # blending bottle scores for a second time.
            for i in range(2):
                path = data / 'iron_lattice/train/good' / f'{i}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(1, 255, (16, 16, 3), dtype=np.uint8)).save(path)
            args.predictions = str(root / 'context')
            args.output_dir = str(root / 'context_iron')
            args.category, args.feature_weights = 'iron_lattice', '0.5,0.5'
            with patch.dict('sys.modules', {'dinomaly_omniad_uni': fake}):
                run(args)
            for preserved in (category, 'wafer2'):
                self.assertEqual((root / 'context' / preserved / 'defect/a.png.npy').read_bytes(),
                                 (root / 'context_iron' / preserved / 'defect/a.png.npy').read_bytes())
            self.assertFalse(np.allclose(np.load(root / 'context_iron/iron_lattice/defect/a.png.npy'), .2))
            with (root / 'context_iron/scores.csv').open(newline='', encoding='utf-8') as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual([r['score'] for r in final_rows], [r['score'] for r in rows])
            args.output_dir = str(root / 'tiled_iron')
            args.tile_grid, args.tile_overlap = 2, .25
            with patch.dict('sys.modules', {'dinomaly_omniad_uni': fake}), \
                    patch('predict_omniad_memory.nearest_distance', wraps=nearest_distance) as retrieval:
                run(args)
            calibration_calls = [call for call in retrieval.call_args_list if len(call.args) == 5]
            self.assertEqual([call.args[4] for call in calibration_calls], [0]*4 + [1]*4)
            tiled = torch.load(root / 'tiled_iron/normal_bank.pt', weights_only=True)
            self.assertEqual(tiled['tile_grid'], 2)
            self.assertEqual(tiled['bank'].shape, (16, 12))
            self.assertEqual(tiled['owners'].tolist(), [0]*8 + [1]*8)
            tiled_map = np.load(root / 'tiled_iron/iron_lattice/defect/a.png.npy')
            self.assertEqual(tiled_map.shape, (16, 16))
            self.assertTrue(np.isfinite(tiled_map).all())
            for preserved in (category, 'wafer2'):
                self.assertEqual((root / 'context' / preserved / 'defect/a.png.npy').read_bytes(),
                                 (root / 'tiled_iron' / preserved / 'defect/a.png.npy').read_bytes())
            with (root / 'tiled_iron/scores.csv').open(newline='', encoding='utf-8') as handle:
                tiled_rows = list(csv.DictReader(handle))
            self.assertEqual([r['score'] for r in tiled_rows], [r['score'] for r in rows])
            # Reuse the same checkpoint mock through the explicit legacy route.
            # Every patch is valid, even for a non-square source view.
            def legacy_preprocess(config, checkpoint):
                config.crop_size, config.image_size, config.preprocess = 16, 20, 'legacy'
            fake.apply_checkpoint_preprocessing = legacy_preprocess
            args.output_dir = str(root / 'legacy')
            args.legacy_full_frame, args.reference_unselected = True, True
            with patch.dict('sys.modules', {'dinomaly_omniad_uni': fake}):
                run(args)
            from export_omniad_routed import read_index
            exported = read_index(root / 'legacy')
            source_records = read_index(root / 'context')
            for preserved in (category, 'wafer2'):
                key = (preserved, 'defect/a.png')
                self.assertEqual(exported[key][1].resolve(), source_records[key][1].resolve())
                self.assertEqual(exported[key][0]['score'], source_records[key][0]['score'])
            artifact = torch.load(root / 'legacy/normal_bank.pt', weights_only=True)
            self.assertEqual(artifact['original_geometry']['image_size'], 20)
            self.assertEqual(artifact['inference_geometry']['image_size'], 16)
            self.assertTrue(np.isfinite(np.load(exported[('iron_lattice', 'defect/a.png')][1])).all())
            for i in range(2, 5):
                image = data / 'iron_lattice/train/good' / f'{i}.png'
                Image.fromarray(rng.integers(1, 255, (16, 16, 3), dtype=np.uint8)).save(image)
            args.output_dir, args.neighbor_images = str(root / 'multi_image'), 3
            with patch.dict('sys.modules', {'dinomaly_omniad_uni': fake}), \
                    patch('predict_omniad_memory.nearest_distance', wraps=nearest_distance) as retrieval:
                run(args)
            self.assertTrue(all(call.kwargs['neighbor_images'] == 3 for call in retrieval.call_args_list))
            calibration_calls = [call for call in retrieval.call_args_list if len(call.args) == 5]
            self.assertEqual([call.args[4] for call in calibration_calls], [i for i in range(5) for _ in range(4)])
            multi = read_index(root / 'multi_image')
            for key in multi:
                self.assertEqual(multi[key][0]['score'], source_records[key][0]['score'])
                if key[0] != 'iron_lattice':
                    self.assertEqual(multi[key][1].resolve(), source_records[key][1].resolve())
            artifact = torch.load(root / 'multi_image/normal_bank.pt', weights_only=True)
            self.assertEqual(artifact['neighbor_images'], 3)


if __name__ == '__main__':
    unittest.main()

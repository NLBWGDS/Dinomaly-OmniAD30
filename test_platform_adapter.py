import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from platform_infer import run as run_inference, visual_objects
from platform_io import contained_path, get_parameter, platform_input_path
from platform_train import learning_rate, run as run_training, stage_training_data


class PlatformAdapterTests(unittest.TestCase):
    def test_parameter_precedence_and_path_containment(self):
        params = {'modelPath': 'outer.bin', 'cnnParam': {'extendParamMap': {'modelPath': 'inner.bin'}}}
        self.assertEqual(get_parameter(params, ('modelPath',)), 'inner.bin')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'inner.bin').touch()
            self.assertEqual(contained_path('inner.bin', root, 'model'), (root / 'inner.bin').resolve())
            model = root / 'model/output.bin'
            model.parent.mkdir()
            model.touch()
            self.assertEqual(platform_input_path('/model/output.bin', root, 'model'), model.resolve())
            self.assertEqual(platform_input_path('/input/model/output.bin', root, 'model'), model.resolve())
            with self.assertRaises(ValueError):
                contained_path('../escape.bin', root, 'model', must_exist=False)

    def test_training_staging_and_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            images = []
            for index, suffix in enumerate(('.png', '.jpg')):
                path = source / f'image{index}{suffix}'
                path.write_bytes(b'image')
                images.append(path)
            staged, category = stage_training_data(images, root / 'runtime')
            files = sorted((staged / category / 'train/good').iterdir())
            self.assertEqual([p.suffix for p in files], ['.png', '.jpg'])
            self.assertAlmostEqual(learning_rate(0, 100, .01, .001, 10), 0.)
            self.assertAlmostEqual(learning_rate(10, 100, .01, .001, 10), .01)
            self.assertAlmostEqual(learning_rate(100, 100, .01, .001, 10), .001)

    def test_inference_protocol_preserves_gt_and_clips_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root, output_root = root / 'input', root / 'output'
            image_root = input_root / 'imgs'
            image_root.mkdir(parents=True)
            image = np.zeros((12, 16, 3), np.uint8)
            Image.fromarray(image).save(image_root / 'sample.png')
            model = input_root / 'models/output.bin'
            model.parent.mkdir()
            model.write_bytes(b'model')
            (input_root / 'pred').mkdir()
            gt = input_root / 'pred/gt.json'
            gt.write_text('{"sentinel": true}', encoding='utf-8')
            params = {'algorithmType': 103, 'algorithmSubType': 0, 'modelType': 1,
                      'modelPath': '/input/models/output.bin',
                      'imagePath': '/input/imgs', 'platType': 1,
                      'cnnParam': {'MinScore': .1, 'pixelThreshold': .2}}
            (input_root / 'param.json').write_text(json.dumps(params, ensure_ascii=False), encoding='utf-8')

            class FakePredictor:
                def __init__(self, checkpoint, device):
                    self.checkpoint, self.device = checkpoint, device

                def __call__(self, path):
                    anomaly = np.zeros((12, 16), np.float32)
                    anomaly[2:10, 3:13] = 1.2
                    return .8, anomaly, 4.25

            # Platform paths in param.json are absolute inside the container.
            params['模型路径'] = str(model)
            params['imagePath'] = str(image_root)
            (input_root / 'param.json').write_text(json.dumps(params, ensure_ascii=False), encoding='utf-8')
            fake_visual = [{'score': .8, 'id': 1, 'labelName': 'NG', 'segmentation': [], 'points': []}]
            with patch('platform_infer.visual_objects', return_value=fake_visual):
                run_inference(SimpleNamespace(input_dir=str(input_root), output_dir=str(output_root)), FakePredictor)
            prediction = json.loads((input_root / 'pred/pred.json').read_text(encoding='utf-8'))
            self.assertEqual(set(prediction), {'test/sample.png'})
            self.assertEqual(prediction['test/sample.png']['anomaly_map'], 'pred_maps/test/sample.npy')
            anomaly = np.load(input_root / 'pred/pred_maps/test/sample.npy')
            self.assertEqual(anomaly.dtype, np.float32)
            self.assertEqual(anomaly.shape, (12, 16))
            self.assertEqual(float(anomaly.max()), 1.)
            objects = json.loads((output_root / 'sample.json').read_text(encoding='utf-8'))
            self.assertEqual(len(objects), 1)
            self.assertIn('segmentation', objects[0])
            self.assertEqual(gt.read_text(encoding='utf-8'), '{"sentinel": true}')
            log = (output_root / 'reasoning.log').read_text(encoding='utf-8')
            self.assertIn('reasoning start', log)
            self.assertIn('reasoning imageName=sample.png,sequence=1', log)
            self.assertRegex(log, r'algRunTime=\d+\.\d+,sdkRunTime=\d+\.\d+')
            self.assertTrue(log.rstrip().endswith('reasoning close success'))

    def test_training_protocol_exports_fixed_model_name_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root, output_root, runtime = root / 'input', root / 'output', root / 'runtime'
            data = input_root / 'original/input_data'
            images = data / 'images/train'
            images.mkdir(parents=True)
            for index in range(2):
                Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(images / f'{index}.png')
            (data / 'param.json').write_text(json.dumps({'total_iters': 2, 'batch_size': 1,
                                                         'num_workers': 0, 'resolution': 560}), encoding='utf-8')

            class FakeProcess:
                stdout = iter(['iter [1/2], loss:0.5000\n', 'iter [2/2], loss:0.2500\n'])

                def wait(self):
                    return 0

            def start(command, **kwargs):
                destination = Path(command[command.index('--output_dir') + 1])
                destination.mkdir(parents=True)
                (destination / 'omniad_dinomaly_uni.pth').write_bytes(b'checkpoint')
                return FakeProcess()

            args = SimpleNamespace(input_dir=str(input_root), output_dir=str(output_root),
                                   runtime_dir=str(runtime), device='cpu')
            with patch('platform_train.subprocess.Popen', side_effect=start), \
                    patch('platform_train.gpu_memory_mb', return_value=321):
                run_training(args)
            self.assertEqual((output_root / 'output.bin').read_bytes(), b'checkpoint')
            state = (output_root / 'state.txt').read_text(encoding='utf-8')
            for keyword in ('Epoch(train)', 'Iter:', 'lr:', 'eta:', 'time:', 'memory:', 'loss:', 'finish'):
                self.assertIn(keyword, state)
            self.assertIn('Epoch(train) [1][1/2] Iter: 1/2', state)
            self.assertIn('memory:321', state)
            self.assertNotIn('finish', '\n'.join(state.splitlines()[:-1]))
            self.assertTrue(state.rstrip().endswith('finish omniad training'))

    def test_visual_empty_below_image_threshold(self):
        self.assertEqual(visual_objects(.09, np.ones((5, 5), np.float32), .1, .2, 0), [])

    def test_visual_schema_matches_unsupervised_polygon(self):
        contour = np.array([[[1, 2]], [[8, 2]], [[8, 9]], [[1, 9]]], np.int32)
        fake_cv2 = SimpleNamespace(
            RETR_EXTERNAL=0, CHAIN_APPROX_SIMPLE=0,
            findContours=lambda binary, mode, method: ([contour], None),
            contourArea=lambda value: 49., arcLength=lambda value, closed: 28.,
            approxPolyDP=lambda value, epsilon, closed: value)
        with patch.dict('sys.modules', {'cv2': fake_cv2}):
            result = visual_objects(.8, np.ones((12, 16), np.float32), .1, .2, 4, '1')
        self.assertEqual(set(result[0]), {'category_Name', 'score', 'id', 'segmentation', 'type'})
        self.assertEqual(result[0]['type'], 'polygon')
        self.assertEqual(result[0]['segmentation'], [[1, 2, 8, 2, 8, 9, 1, 9]])


if __name__ == '__main__':
    unittest.main()

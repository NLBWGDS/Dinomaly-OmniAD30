"""Competition inference entry implementing /input and /output protocols."""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from platform_io import (atomic_json, bounded_number, discover_images,
                         get_parameter, load_json, platform_input_path)


class ReasoningLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open('w', encoding='utf-8', buffering=1)

    def write(self, message):
        print(message, flush=True)
        self.handle.write(message + '\n')

    def close(self):
        self.handle.close()


class DinomalyPredictor:
    def __init__(self, checkpoint_path, device):
        import torch
        from dinomaly_omniad_uni import (apply_checkpoint_preprocessing, get_omniad_transforms,
                                        load_checkpoint_model)
        from utils import get_gaussian_kernel

        self.torch = torch
        self.device = torch.device(device)
        config = SimpleNamespace(checkpoint=str(checkpoint_path), encoder='dinov2reg_vit_base_14',
                                 image_size=560, crop_size=560, eval_mask_size=560,
                                 preprocess='letterbox', train_augment=False)
        self.model, checkpoint = load_checkpoint_model(config, self.device)
        apply_checkpoint_preprocessing(config, checkpoint)
        self.transform, _ = get_omniad_transforms(config)
        self.config = config
        self.kernel = get_gaussian_kernel(
            checkpoint.get('gaussian_kernel_size', 3), checkpoint.get('gaussian_sigma', 1.)
        ).to(self.device)

    def __call__(self, image_path):
        from PIL import Image
        from dinomaly_omniad_uni import restore_anomaly_map, score_anomaly_map
        from utils import cal_anomaly_maps

        torch = self.torch
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            width, height = image.size
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        sdk_start = time.perf_counter()
        with torch.inference_mode():
            encoded, decoded = self.model(tensor)
            anomaly, _ = cal_anomaly_maps(encoded, decoded, tensor.shape[-1], feature_weights=[.5, .5])
            anomaly = self.kernel(anomaly)
            score = score_anomaly_map(anomaly, height, width, SimpleNamespace(
                preprocess=self.config.preprocess, max_ratio=.01))
            restored = restore_anomaly_map(anomaly, height, width, self.config)[0, 0]
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        sdk_ms = (time.perf_counter() - sdk_start) * 1000
        return float(score), restored.float().cpu().numpy(), sdk_ms


def visual_objects(score, anomaly_map, min_score, pixel_threshold, min_area, category_name='1'):
    if score < min_score:
        return []
    import cv2

    binary = (anomaly_map >= pixel_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(1., .002 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        flat_polygon = [int(value) for point in polygon for value in point]
        objects.append({'category_Name': str(category_name), 'score': float(score),
                        'id': len(objects)+1, 'segmentation': [flat_polygon], 'type': 'polygon'})
    return objects


def run(args, predictor_factory=DinomalyPredictor):
    input_root, output_root = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log = ReasoningLog(output_root / 'reasoning.log')
    log.write('reasoning start')
    try:
        params = load_json(input_root / 'param.json')
        algorithm_type = get_parameter(params, ('algorithmType', '算法类型'))
        if str(algorithm_type) not in {'103', '无监督分割'}:
            raise ValueError(f'algorithmType must be 103 (unsupervised segmentation), got {algorithm_type}')
        subtype = get_parameter(params, ('algorithmSubType', '算法子类型'), 0)
        if str(subtype) != '0':
            raise ValueError(f'algorithmSubType must be 0, got {subtype}')
        model_type = get_parameter(params, ('modelType', '模型类型'))
        if str(model_type) not in {'1', '4'}:
            raise ValueError(f'modelType must be 1 or 4, got {model_type}')
        password = get_parameter(params, ('模型加密密码', 'modelPassword', 'password'))
        if password not in (None, ''):
            raise ValueError('encrypted model files are not supported by this image')
        model_raw = get_parameter(params, ('模型路径', 'modelPath', 'model_path'))
        image_raw = get_parameter(params, ('imagePath', '图片路径', 'image_path'))
        if model_raw is None or image_raw is None:
            raise ValueError('param.json requires modelPath/模型路径 and imagePath')
        model_path = platform_input_path(model_raw, input_root, 'modelPath')
        image_path = platform_input_path(image_raw, input_root, 'imagePath')
        if not model_path.is_file():
            raise ValueError('modelPath must be a file')
        images = discover_images(image_path)
        platform = str(get_parameter(params, ('platType', 'platTyppe', '推理平台类型'), '2'))
        if platform not in {'1', '2'}:
            raise ValueError('platType must be 1 (CPU) or 2 (GPU)')
        device = 'cpu' if platform == '1' else 'cuda:0'
        if platform == '2':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('platType=2 requested but CUDA is unavailable')
        min_score = bounded_number(params, ('MinScore', 'minScore', 'confidence'), .1, float, 0., 1.)
        pixel_threshold = bounded_number(params, ('pixelThreshold', '像素阈值'), .12, float, 0., 1.)
        min_area = bounded_number(params, ('minArea', '最小面积'), 4, int, 0, 1000000000)
        category_name = get_parameter(params, ('category_Name', 'categoryName', 'labelName'), '1')
        predictor = predictor_factory(model_path, device)
        pred_root = input_root / 'pred'
        map_root = pred_root / 'pred_maps' / 'test'
        map_root.mkdir(parents=True, exist_ok=True)
        predictions = {}
        for sequence, image in enumerate(images, 1):
            started = time.perf_counter()
            score, anomaly_map, sdk_ms = predictor(image)
            anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
            from PIL import Image
            with Image.open(image) as source:
                expected_shape = (source.height, source.width)
            if anomaly_map.shape != expected_shape or not np.isfinite(anomaly_map).all():
                raise ValueError(f'invalid anomaly map for {image.name}: {anomaly_map.shape}')
            normalized = np.clip(anomaly_map, 0., 1.).astype(np.float32)
            bounded_score = float(np.clip(score, 0., 1.))
            relative_map = f'pred_maps/test/{image.stem}.npy'
            map_path = pred_root / relative_map
            temporary = map_path.with_name(map_path.name + '.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, normalized, allow_pickle=False)
            os.replace(temporary, map_path)
            key = f'test/{image.name}'
            predictions[key] = {'anomaly_score': bounded_score, 'anomaly_map': relative_map}
            atomic_json(output_root / f'{image.stem}.json',
                        visual_objects(bounded_score, normalized, min_score, pixel_threshold,
                                       min_area, category_name))
            alg_ms = (time.perf_counter() - started) * 1000
            log.write(f'reasoning imageName={image.name},sequence={sequence},'
                      f'algRunTime={alg_ms:.6f},sdkRunTime={sdk_ms:.6f}')
        atomic_json(pred_root / 'pred.json', predictions)
        log.write('reasoning close success')
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'.replace('\r', ' ').replace('\n', ' ')
        log.write(f'reasoning error, code=0x80100000, message={message}')
        raise
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='/input')
    parser.add_argument('--output_dir', default='/output')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

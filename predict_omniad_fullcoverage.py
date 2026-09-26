"""Fill unseen borders of legacy center-crop inference, preserving center scores.

Uses the original resize scale and crop size. No anomaly labels are consumed.
All image-level scores and all unselected maps are retained from the baseline.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate_omniad_all30 import dataset_inventory, require_exact_keys
from export_omniad_routed import read_index
from predict_omniad_tiled import MapStitcher
from omniad_normal_calibration import fit_normal_calibration, calibrate_border


def coverage_boxes(side, crop):
    if not isinstance(side, int) or not isinstance(crop, int) or not 0 < crop <= side:
        raise ValueError('Require integer 0 < crop_size <= image_size')
    offset = int(round((side - crop) / 2.))
    center = (offset, offset, offset + crop, offset + crop)
    steps = math.ceil((side - crop) / crop) + 1
    starts = sorted(set(round(v) for v in np.linspace(0, side - crop, steps)))
    return list(dict.fromkeys([center] + [(x, y, x + crop, y + crop) for y in starts for x in starts]))


class CoverageStitcher:
    def __init__(self, side, center_box):
        self.stitch = MapStitcher(side, side)
        self.center_box = center_box
        self.center = None

    def add(self, box, scores):
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite crop map')
        self.stitch.add(box, scores)
        if box == self.center_box:
            self.center = scores.copy()

    def finish(self):
        if self.center is None:
            raise ValueError('Missing original center-crop prediction')
        result = self.stitch.finish()
        x0, y0, x1, y1 = self.center_box
        result[y0:y1, x0:x1] = self.center
        return result


def coverage_canvas(tensor, crop_size, batch_size, infer_batch):
    if tensor.ndim != 3 or tensor.shape[-2] != tensor.shape[-1] or batch_size < 1:
        raise ValueError('Expected square CHW tensor and positive batch size')
    boxes = coverage_boxes(tensor.shape[-1], crop_size)
    stitch = CoverageStitcher(tensor.shape[-1], boxes[0])
    for offset in range(0, len(boxes), batch_size):
        subset = boxes[offset:offset + batch_size]
        batch = torch.stack([tensor[:, y0:y1, x0:x1] for x0, y0, x1, y1 in subset])
        scores = infer_batch(batch)
        if scores.shape != (len(subset), crop_size, crop_size):
            raise ValueError('Inference must return one crop-sized map per input window')
        for box, score in zip(subset, scores):
            stitch.add(box, score)
    return stitch.finish()


def export_predictions(records, expected, selected, output, predict_map):
    """Reference unchanged maps without copying large all30 exports."""
    require_exact_keys(records, expected, 'Full baseline')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category', 'image_path', 'score', 'map_path'])
        completed = 0
        for (category, relative), image_path in sorted(expected.items()):
            row, original_map = records[(category, relative)]
            if not np.isfinite(float(row['score'])):
                raise ValueError('Nonfinite baseline image score')
            if category in selected:
                result = predict_map(image_path)
                with Image.open(image_path) as image:
                    shape = (image.height, image.width)
                if result.shape != shape or not np.isfinite(result).all():
                    raise ValueError(f'Invalid full-coverage map for {image_path}')
                target = output / category / (relative + '.npy')
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, result.astype(np.float32))
                completed += 1
                if completed % 20 == 0:
                    print(f'Full coverage: {completed} images exported', flush=True)
            else:
                target = original_map
            writer.writerow([category, str(image_path), row['score'], str(target.resolve())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='./diagnostics/all30_current/current_predictions')
    parser.add_argument('--checkpoint', default='./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth')
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--output_dir', default='./predictions/all30_fullcoverage')
    parser.add_argument('--categories', default='nameplate7,ceramic_wafer,spindle_top,chip1')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--normal_calibration', action='store_true',
                        help='Fit border score calibration from train/good only; preserve center scores.')
    parser.add_argument('--calibration_grid', type=int, default=32)
    parser.add_argument('--calibration_strength', type=float, default=.5)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch_size must be positive')
    if args.calibration_grid < 2 or not np.isfinite(args.calibration_strength) or not 0 <= args.calibration_strength <= 1:
        parser.error('Invalid calibration grid or strength')
    categories, expected = dataset_inventory(args.data_path)
    selected = {c.strip() for c in args.categories.split(',') if c.strip()}
    if not selected or not selected <= set(categories):
        raise ValueError('Selected categories must be present in the all30 dataset')
    records = read_index(Path(args.predictions))
    require_exact_keys(records, expected, 'Full baseline')
    if Path(args.output_dir).exists():
        raise FileExistsError('Use a new output directory')
    from dinomaly_omniad_uni import (load_checkpoint_model, apply_checkpoint_preprocessing,
                                    IMAGENET_MEAN, IMAGENET_STD)
    from torchvision.transforms import functional as TF, InterpolationMode
    from utils import cal_anomaly_maps, get_gaussian_kernel

    device = torch.device(args.device)
    config = SimpleNamespace(checkpoint=args.checkpoint, encoder='dinov2reg_vit_base_14',
                             image_size=560, crop_size=560)
    model, checkpoint = load_checkpoint_model(config, device)
    if not selected <= set(checkpoint.get('categories', [])):
        raise ValueError('Checkpoint category metadata does not cover selected categories')
    apply_checkpoint_preprocessing(config, checkpoint)
    if config.preprocess != 'legacy' or config.crop_size >= config.image_size:
        raise ValueError('This experiment requires legacy center cropping with crop_size < image_size; '
                         'the supplied checkpoint does not have the hypothesized missing border')
    boxes = coverage_boxes(config.image_size, config.crop_size)
    kernel = get_gaussian_kernel(3, 1.).to(device)
    missing_fraction = 1 - (config.crop_size / config.image_size)**2
    print(f'Legacy resize={config.image_size}, crop={config.crop_size}; '
          f'previously unseen area={missing_fraction:.2%}; views/image={len(boxes)}', flush=True)
    timings = []

    @torch.inference_mode()
    def make_canvas(path):
        with Image.open(path) as image:
            width, height = image.size
            resized = TF.resize(image.convert('RGB'), [config.image_size, config.image_size],
                                interpolation=InterpolationMode.BILINEAR)
            tensor = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
        def infer_batch(batch):
            en, de = model(batch.to(device))
            maps, _ = cal_anomaly_maps(en, de, config.crop_size, feature_weights=[.5, .5])
            return kernel(maps)[:, 0].cpu().numpy()
        return coverage_canvas(tensor, config.crop_size, args.batch_size, infer_batch), (height, width)

    calibrations, calibration_paths = {}, {}
    calibration_start = time.perf_counter()
    if args.normal_calibration:
        if args.calibration_grid > config.image_size:
            raise ValueError('Calibration grid exceeds resized canvas size')
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        for category in sorted(selected):
            root = Path(args.data_path) / category / 'train' / 'good'
            paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
            if len(paths) < 5:
                raise ValueError(f'{category}: need at least five train/good images')
            calibration_paths[category] = [str(p.resolve()) for p in paths]
            def normal_maps():
                for index, path in enumerate(paths):
                    canvas, _ = make_canvas(path)
                    if (index+1) % 10 == 0 or index+1 == len(paths):
                        print(f'{category}: normal calibration {index+1}/{len(paths)}', flush=True)
                    yield canvas
            calibrations[category] = fit_normal_calibration(normal_maps(), config.image_size,
                                                            boxes[0], args.calibration_grid)
    calibration_seconds = time.perf_counter() - calibration_start
    category_by_path = {str(p.resolve()): key[0] for key, p in expected.items()}

    def predict_map(path):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        canvas, (height, width) = make_canvas(path)
        category = category_by_path[str(path.resolve())]
        if category in calibrations:
            canvas = calibrate_border(canvas, calibrations[category], args.calibration_strength)
        canvas = torch.from_numpy(canvas)[None, None]
        result = F.interpolate(canvas, size=(height, width), mode='bilinear', align_corners=False)[0, 0].numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        return result

    export_predictions(records, expected, selected, args.output_dir, predict_map)
    for category, calibration in calibrations.items():
        np.savez_compressed(Path(args.output_dir) / f'{category}_normal_calibration.npz', **calibration)
    manifest = dict(arguments=vars(args), selected_categories=sorted(selected),
                    preprocess=config.preprocess, image_size=config.image_size, crop_size=config.crop_size,
                    previously_unseen_area_fraction=missing_fraction, windows=boxes,
                    fusion='Keep original center window; weighted shifted-window scores fill outside only',
                    image_scores='Copied unchanged from baseline CSV',
                    labels_used=False, mean_branch_ms=float(np.mean(timings)),
                    normal_calibration=args.normal_calibration, calibration_images=calibration_paths,
                    calibration_seconds=calibration_seconds,
                    note='Experimental coverage ablation, not proven improvement. '
                         'References unselected baseline maps; preserve source directories. '
                         'Timing excludes export and prior baseline inference, no warmup.')
    (Path(args.output_dir) / 'coverage_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Exported all30 predictions to {args.output_dir}; {len(timings)} selected images recomputed')


if __name__ == '__main__':
    main()

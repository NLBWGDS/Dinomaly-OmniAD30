"""Overlap-tiled inference using an existing Omni-AD checkpoint."""

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

def tile_boxes(width, height, grid, overlap):
    if grid < 1 or not 0 <= overlap < 1:
        raise ValueError('grid must be positive and overlap must be in [0, 1)')
    tw = min(width, math.ceil(width / (grid - (grid - 1) * overlap)))
    th = min(height, math.ceil(height / (grid - (grid - 1) * overlap)))
    xs = sorted(set(round(v) for v in np.linspace(0, width - tw, grid)))
    ys = sorted(set(round(v) for v in np.linspace(0, height - th, grid)))
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


class MapStitcher:
    def __init__(self, width, height):
        self.total = np.zeros((height, width), dtype=np.float32)
        self.weight = np.zeros_like(self.total)

    def add(self, box, scores):
        x0, y0, x1, y1 = box
        if scores.shape != (y1 - y0, x1 - x0):
            raise ValueError('Tile scores do not match crop geometry')
        # Nonzero edge weights cover the outer image boundary as well.
        wy = np.hanning(scores.shape[0] + 2)[1:-1]
        wx = np.hanning(scores.shape[1] + 2)[1:-1]
        weight = np.maximum(np.outer(wy, wx), 0.01).astype(np.float32)
        self.total[y0:y1, x0:x1] += scores * weight
        self.weight[y0:y1, x0:x1] += weight

    def finish(self):
        if np.any(self.weight <= 0):
            raise ValueError('Uncovered pixels in tiled inference')
        return self.total / self.weight


def main():
    from dinomaly_omniad_uni import (
        parse_args, validate_args, discover_categories, load_checkpoint_model,
        apply_checkpoint_preprocessing, get_omniad_transforms, get_feature_weights,
        restore_anomaly_map,
    )
    from utils import cal_anomaly_maps, get_gaussian_kernel

    args = parse_args()
    validate_args(args)
    if args.mode != 'predict' or not args.checkpoint:
        raise ValueError('Use --mode predict and --checkpoint')
    if not 0 <= args.global_weight <= 1:
        raise ValueError('--global_weight must be in [0, 1]')
    tile_boxes(100, 100, args.tile_grid, args.tile_overlap)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_checkpoint_model(args, device)
    apply_checkpoint_preprocessing(args, checkpoint)
    if args.preprocess != 'letterbox':
        raise ValueError('Tiled inference requires a letterbox checkpoint to preserve crop coverage')
    transform, _ = get_omniad_transforms(args)
    gaussian = get_gaussian_kernel(args.gaussian_kernel_size, args.gaussian_sigma).to(device)
    output = Path(args.output_dir)
    if (output / 'scores.csv').exists():
        raise FileExistsError('Choose a new output directory to preserve existing predictions')
    output.mkdir(parents=True, exist_ok=True)
    timings = []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    with torch.inference_mode(), (output / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category', 'image_path', 'score', 'map_path'])
        for category in discover_categories(args.data_path, args.categories):
            root = Path(args.data_path) / category / 'test'
            paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in extensions)
            if not paths:
                raise ValueError(f'No test images in {root}')
            destination = output / category
            destination.mkdir(exist_ok=True)
            for index, path in enumerate(paths):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with Image.open(path) as source:
                    image = source.convert('RGB')
                boxes = tile_boxes(image.width, image.height, args.tile_grid, args.tile_overlap)
                jobs = boxes + ([(0, 0, image.width, image.height)] if args.global_weight > 0 else [])
                stitch = MapStitcher(image.width, image.height)
                global_map = None
                for offset in range(0, len(jobs), args.batch_size):
                    subset = jobs[offset:offset + args.batch_size]
                    batch = torch.stack([transform(image.crop(box)) for box in subset]).to(device)
                    en, de = model(batch)
                    maps, _ = cal_anomaly_maps(en, de, batch.shape[-1],
                                              feature_weights=get_feature_weights(args, category))
                    maps = gaussian(maps)
                    for j, box in enumerate(subset):
                        score_map = restore_anomaly_map(maps[j:j+1], box[3]-box[1], box[2]-box[0], args)
                        score_map = score_map[0, 0].cpu().numpy()
                        if offset + j < len(boxes):
                            stitch.add(box, score_map)
                        else:
                            global_map = score_map
                result = stitch.finish()
                if global_map is not None:
                    result = (1 - args.global_weight) * result + args.global_weight * global_map
                flat = result.ravel()
                k = max(1, int(flat.size * args.max_ratio))
                score = float(np.partition(flat, flat.size-k)[-k:].mean())
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                timings.append(1000 * (time.perf_counter() - start))
                # Keep the relative image filename, including extension, to avoid collisions.
                name = '__'.join(path.relative_to(root).parts) + '.npy'
                map_path = destination / name
                np.save(map_path, result)
                writer.writerow([category, str(path.resolve()), score, str(map_path.resolve())])
                if (index + 1) % 10 == 0 or index == len(paths) - 1:
                    print(f'{category}: {index+1}/{len(paths)}', flush=True)
    (output / 'inference_config.json').write_text(json.dumps(dict(
        arguments=vars(args), checkpoint_preprocess=args.preprocess,
        mean_ms=float(np.mean(timings)), p95_ms=float(np.percentile(timings, 95)),
        timing_note='Includes decode, preprocessing, inference and fusion; excludes export. No warmup.',
    ), indent=2), encoding='utf-8')
    print(f'Predictions saved to {output}; mean {np.mean(timings):.1f} ms/image')


if __name__ == '__main__':
    main()

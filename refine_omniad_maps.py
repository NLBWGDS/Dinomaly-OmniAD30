"""Label-free, image-guided refinement of exported pixel maps.

Grayscale guided filter: He et al., Guided Image Filtering, ECCV 2010.
https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf
Image-level scores are copied unchanged, independently of pixel refinement.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from export_omniad_routed import read_index


def validate_parameters(radius, eps, strength):
    if not isinstance(radius, int) or radius < 1:
        raise ValueError('radius must be a positive integer')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be in [0, 1]')


def refine_map(scores, guide, radius=8, eps=0.001, strength=0.5):
    validate_parameters(radius, eps, strength)
    p, g = np.asarray(scores, dtype=np.float64), np.asarray(guide, dtype=np.float64)
    if p.ndim != 2 or p.shape != g.shape or not p.size:
        raise ValueError('Map and grayscale guide must have identical nonempty 2D shapes')
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Nonfinite map or guide')
    if g.min() < 0 or g.max() > 1:
        raise ValueError('Guide must be scaled to [0, 1]')
    if strength == 0 or p.min() == p.max():
        return p.astype(np.float32)

    def mean(x):
        return uniform_filter(x, size=2 * radius + 1, mode='reflect')

    mg, mp = mean(g), mean(p)
    variance = np.maximum(mean(g * g) - mg * mg, 0)
    covariance = mean(g * p) - mg * mp
    a = covariance / (variance + eps)
    b = mp - a * mg
    guided = mean(a) * g + mean(b)
    # Bound linear-model overshoot; do not normalize each image independently.
    guided = np.clip(guided, p.min(), p.max())
    return ((1 - strength) * p + strength * guided).astype(np.float32)


def export_refined(predictions, data, output, categories=('wafer2',),
                   radius=8, eps=0.001, strength=0.5):
    validate_parameters(radius, eps, strength)
    records = read_index(Path(predictions))
    selected = set(categories)
    if not records or not selected or not selected <= {k[0] for k in records}:
        raise ValueError('Predictions must be nonempty and contain all selected categories')
    data, output = Path(data), Path(output)
    # Validate inputs before creating the destination. No masks are opened.
    for (category, relative), (row, _) in records.items():
        if not np.isfinite(float(row['score'])):
            raise ValueError('Nonfinite image score')
        if category in selected and not (data / category / 'test' / relative).is_file():
            raise FileNotFoundError(data / category / 'test' / relative)
    output.mkdir(parents=True, exist_ok=False)
    elapsed, changed = [], 0
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category', 'image_path', 'score', 'map_path'])
        for (category, relative), (row, map_path) in records.items():
            target = output / category / (relative + '.npy')
            target.parent.mkdir(parents=True, exist_ok=True)
            if category in selected:
                start = time.perf_counter()
                scores = np.load(map_path, allow_pickle=False)
                with Image.open(data / category / 'test' / relative) as image:
                    guide = np.asarray(image.convert('L'), dtype=np.float64) / 255.
                result = refine_map(scores, guide, radius, eps, strength)
                np.save(target, result)
                elapsed.append((time.perf_counter() - start) * 1000)
                changed += 1
            else:
                shutil.copyfile(map_path, target)
            writer.writerow([category, row['image_path'], row['score'], str(target.resolve())])
            if changed and changed % 50 == 0 and category in selected:
                print(f'Refined {changed} maps', flush=True)
    manifest = dict(source=str(predictions), categories=sorted(selected), radius=radius,
                    eps=eps, strength=strength, refined_images=changed,
                    mean_refinement_ms=float(np.mean(elapsed)),
                    image_scores='Copied verbatim from source scores.csv',
                    labels_used=False,
                    note='Experimental boundary refinement, not proven improvement. '
                         'Timing includes image/map IO; radius is in original-image pixels.')
    (output / 'refinement_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Exported {len(records)} predictions; refined {changed}; image scores unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--categories', default='wafer2')
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--strength', type=float, default=0.5)
    args = parser.parse_args()
    export_refined(args.predictions, args.data_path, args.output_dir,
                   [c.strip() for c in args.categories.split(',') if c.strip()],
                   args.radius, args.eps, args.strength)

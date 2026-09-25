"""Development-only error analysis of exported, original-resolution maps."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def locate(row, predictions, data):
    category = row['category']
    if Path(category).name != category:
        raise ValueError('Invalid category name')
    parts = row['image_path'].replace('\\', '/').split('/')
    marker = next((i for i in range(len(parts) - 1)
                   if parts[i] == category and parts[i + 1] == 'test'), None)
    if marker is None:
        raise ValueError(f"Cannot identify category/test in {row['image_path']}")
    relative = Path(*parts[marker + 2:])
    if '..' in relative.parts or len(relative.parts) < 2:
        raise ValueError(f'Invalid test path: {relative}')
    image = data / category / 'test' / relative
    exported = Path(row['map_path'])
    candidates = [exported, predictions / category / row['map_path'].replace('\\', '/').split('/')[-1]]
    maps = next((p for p in candidates if p.is_file()), None)
    if maps is None:
        raise FileNotFoundError(row['map_path'])
    mask = None
    if relative.parts[0] != 'good':
        folder = data / category / 'ground_truth' / relative.parent
        matches = [p for p in folder.glob('*')
                   if p.stem in (relative.stem, relative.stem + '_mask')]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one mask for {image}; found {matches}')
        mask = matches[0]
    return image, maps, mask


def read_pair(paths):
    image, maps, mask = paths
    with Image.open(image) as im:
        shape = (im.height, im.width)
    scores = np.load(maps, allow_pickle=False)
    if scores.shape != shape or not np.isfinite(scores).all():
        raise ValueError(f'Invalid map shape/values for {image}: {scores.shape}, expected {shape}')
    if mask is None:
        truth = np.zeros(shape, dtype=bool)
    else:
        with Image.open(mask) as im:
            truth = np.asarray(im)
        if truth.ndim == 3:
            truth = truth.max(axis=2)
        if truth.shape != shape:
            raise ValueError(f'Mask/image coordinates differ for {image}')
        truth = truth > 0
    return scores, truth


def counts(scores, truth, threshold):
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def choose_threshold(positive, negative, edges):
    tp = positive[::-1].cumsum()[::-1]
    fp = negative[::-1].cumsum()[::-1]
    fn = positive.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(edges[int(np.argmax(f1))])


def panel(paths, threshold, low, high, destination):
    scores, truth = read_pair(paths)
    with Image.open(paths[0]) as im:
        rgb = np.asarray(im.convert('RGB')).copy()
    predicted = scores >= threshold
    heat = np.clip((scores - low) / max(high - low, 1e-12), 0, 1)
    heat_rgb = np.stack([heat, np.zeros_like(heat), 1 - heat], axis=-1) * 255
    overlay = (0.6 * rgb + 0.4 * heat_rgb).astype(np.uint8)
    error = (rgb * 0.25).astype(np.uint8)
    error[predicted & truth] = (0, 220, 0)
    error[predicted & ~truth] = (255, 50, 50)
    error[~predicted & truth] = (50, 130, 255)
    mask = np.repeat((truth * 255).astype(np.uint8)[..., None], 3, axis=2)
    width = 440
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    canvas = Image.new('RGB', (width * 4, height + 35), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(zip(
            ['Image', 'Ground truth', 'Score (category scale)', 'Green TP / Red FP / Blue FN'],
            [rgb, mask, overlay, error])):
        view = Image.fromarray(array).resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(view, (i * width, 35))
        draw.text((i * width + 5, 8), label, fill='black')
    canvas.save(destination)


def run(args):
    predictions, data, output = Path(args.predictions), Path(args.data_path), Path(args.output_dir)
    with (predictions / 'scores.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    categories = args.categories.split(',') if args.categories else sorted({r['category'] for r in rows})
    summary = {}
    for category in categories:
        selected = [r for r in rows if r['category'] == category]
        if not selected:
            raise ValueError(f'No predictions for {category}')
        paths = [locate(r, predictions, data) for r in selected]
        if len({str(p[0]) for p in paths}) != len(paths):
            raise ValueError(f'Duplicate images for {category}')
        low, high = float('inf'), float('-inf')
        for p in paths:
            scores, _ = read_pair(p)
            low, high = min(low, float(scores.min())), max(high, float(scores.max()))
        edges = np.linspace(low, high if high > low else low + 1e-6, args.bins + 1)
        positive = np.zeros(args.bins, dtype=np.int64)
        negative = np.zeros_like(positive)
        for p in paths:
            scores, truth = read_pair(p)
            positive += np.histogram(scores[truth], bins=edges)[0]
            negative += np.histogram(scores[~truth], bins=edges)[0]
        if not positive.sum():
            raise ValueError(f'No positive pixels for {category}; cannot diagnose anomaly recall')
        threshold = choose_threshold(positive, negative, edges)
        records = []
        for row, p in zip(selected, paths):
            scores, truth = read_pair(p)
            records.append(dict(image_path=str(p[0]), image_score=float(row['score']),
                                threshold=threshold, **counts(scores, truth, threshold)))
        folder = output / category
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        for criterion in ('fp', 'fn'):
            order = sorted(range(len(records)), key=lambda i: records[i][criterion], reverse=True)
            for rank, index in enumerate(order[:args.top_k], 1):
                panel(paths[index], threshold, low, high, folder / f'{criterion}_{rank:02d}.png')
        tp, fp, fn = (sum(r[k] for r in records) for k in ('tp', 'fp', 'fn'))
        summary[category] = dict(images=len(records), tp=tp, fp=fp, fn=fn,
                                 pixel_precision=tp / max(tp + fp, 1),
                                 pixel_recall=tp / max(tp + fn, 1),
                                 pixel_f1=2 * tp / max(2 * tp + fp + fn, 1),
                                 diagnostic_threshold=threshold)
        print(category, json.dumps(summary[category]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(protocol='Original-resolution maps; category histogram threshold search on labeled data. '
                           'Diagnostic only; not a deployable threshold or the legacy evaluation protocol.',
                  bins=args.bins, categories=summary)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--output_dir', default='./diagnostics/hard3')
    parser.add_argument('--categories')
    parser.add_argument('--bins', type=int, default=4096)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    if args.bins < 2 or args.top_k < 0:
        parser.error('bins must be >=2 and top_k >=0')
    run(args)

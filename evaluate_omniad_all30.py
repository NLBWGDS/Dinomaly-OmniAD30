"""Audit all categories: fresh unified-model baseline versus fixed hard3 overrides.

This is a development evaluator, not a single-model submission implementation.
Prediction manifests reference source heatmaps; do not remove source directories.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from export_omniad_routed import read_index


HARD3 = {'infusion_bottle_bottom5', 'iron_lattice', 'wafer2'}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
METRICS = ('pixel_f1', 'image_f1', 'pixel_aupro', 'priority_objective')


def dataset_inventory(data, expected_categories=30):
    data = Path(data)
    categories = sorted(p.name for p in data.iterdir() if p.is_dir() and (p / 'train/good').is_dir())
    if len(categories) != expected_categories:
        raise ValueError(f'Expected {expected_categories} categories, found {len(categories)}: {categories}')
    expected = {}
    for category in categories:
        root = data / category / 'test'
        images = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f'No evaluation images in {root}')
        expected.update({(category, p.relative_to(root).as_posix()): p.resolve() for p in images})
    return categories, expected


def require_exact_keys(records, expected, name):
    missing, extra = set(expected) - set(records), set(records) - set(expected)
    if missing or extra:
        raise ValueError(f'{name}: missing={len(missing)} {sorted(missing)[:5]}, '
                         f'extra={len(extra)} {sorted(extra)[:5]}')
    for key, (row, _) in records.items():
        if not math.isfinite(float(row['score'])):
            raise ValueError(f'{name}: nonfinite image score for {key}')


def combine_predictions(baseline, overrides, expected, output, override_categories=HARD3):
    base = read_index(Path(baseline))
    override = read_index(Path(overrides))
    require_exact_keys(base, expected, 'Unified baseline')
    selected = {key: image for key, image in expected.items() if key[0] in override_categories}
    if {k[0] for k in selected} != set(override_categories):
        raise ValueError('Dataset must contain every configured override category')
    require_exact_keys(override, selected, 'Accepted overrides')
    # A heatmap cannot silently stand in for multiple different images.
    for name, records in [('baseline', base), ('overrides', override)]:
        paths = [path.resolve() for _, path in records.values()]
        if len(paths) != len(set(paths)):
            raise ValueError(f'{name}: duplicate heatmap paths for distinct images')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category', 'image_path', 'score', 'map_path'])
        for key, image in sorted(expected.items()):
            row, path = (override if key[0] in override_categories else base)[key]
            writer.writerow([key[0], str(image), row['score'], str(path.resolve())])
    sources = {category: str(Path(overrides if category in override_categories else baseline).resolve())
               for category in sorted({key[0] for key in expected})}
    manifest = dict(category_sources=sources, images=len(expected),
                    override_categories=sorted(override_categories),
                    note='Fixed category-routed multi-checkpoint candidate, NOT single-model evaluation. '
                         'CSV references existing maps without copying them. Preserve source directories. '
                         'Scores are copied unchanged; no selection uses evaluation labels.')
    (output / 'sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return sources


def prediction_command(args, categories, destination):
    return [sys.executable, '-u', str(Path(__file__).with_name('dinomaly_omniad_uni.py')),
            '--mode', 'predict', '--data_path', str(Path(args.data_path).resolve()),
            '--categories', ','.join(categories), '--checkpoint', str(Path(args.checkpoint).resolve()),
            '--output_dir', str(Path(destination).resolve()), '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers), '--device', args.device,
            '--feature_weights', '0.5,0.5', '--gaussian_kernel_size', '3',
            '--gaussian_sigma', '1.0', '--max_ratio', '0.01']


def write_comparison(baseline, current, sources, destination):
    if baseline['image_ids'] != current['image_ids'] or baseline['protocol'] != current['protocol']:
        raise ValueError('Image list or metric protocol differs between evaluations')
    if baseline['categories'].keys() != current['categories'].keys():
        raise ValueError('Category sets differ')
    fields = ['category', 'images', 'prediction_source']
    for metric in METRICS:
        fields.extend(['baseline_' + metric, 'current_' + metric, 'delta_' + metric])
    with Path(destination).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        ordered = sorted(current['categories'], key=lambda c: (
            current['categories'][c]['pixel_f1'] is None,
            current['categories'][c]['pixel_f1'] or 0, c))
        for category in ordered + ['Mean']:
            a = baseline['mean'] if category == 'Mean' else baseline['categories'][category]
            b = current['mean'] if category == 'Mean' else current['categories'][category]
            row = dict(category=category, images='' if category == 'Mean' else b['images'],
                       prediction_source='macro average' if category == 'Mean' else sources[category])
            for metric in METRICS:
                row['baseline_' + metric], row['current_' + metric] = a[metric], b[metric]
                row['delta_' + metric] = None if a[metric] is None or b[metric] is None else b[metric] - a[metric]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--checkpoint', default='./saved_results/omniad_dinomaly_uni/omniad_dinomaly_uni.pth')
    parser.add_argument('--overrides', default='./predictions/hard3_memory_context')
    parser.add_argument('--output_dir', default='./diagnostics/all30_current')
    parser.add_argument('--baseline_predictions', help='Reuse a complete all30 export instead of new inference')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bins', type=int, default=4096)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.bins < 2:
        parser.error('Invalid batch_size, num_workers or bins')
    categories, expected = dataset_inventory(args.data_path)
    accepted = read_index(Path(args.overrides))
    require_exact_keys(accepted, {k: v for k, v in expected.items() if k[0] in HARD3}, 'Accepted hard3')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new output directory to preserve previous reports')
    if not args.baseline_predictions:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        trained = set(checkpoint.get('categories', []))
        if not set(categories) <= trained:
            raise ValueError('Checkpoint metadata does not cover all30. Do not substitute a hard3 checkpoint.')
        del checkpoint
    baseline = Path(args.baseline_predictions) if args.baseline_predictions else output / 'unified_predictions'
    output.mkdir(parents=True)
    command = None
    if not args.baseline_predictions:
        command = prediction_command(args, categories, baseline)
        print('Generating fresh all30 unified-model predictions...', flush=True)
        subprocess.run(command, check=True)
    sources = combine_predictions(baseline, args.overrides, expected, output / 'current_predictions')
    from evaluate_omniad_export import evaluate
    reports = []
    for name, predictions in [('unified_all30', baseline), ('current_all30', output / 'current_predictions')]:
        print(f'Evaluating {name} at original image resolution...', flush=True)
        report = evaluate(predictions, args.data_path, args.bins)
        report['category_sources'] = sources if name == 'current_all30' else {c: str(baseline.resolve()) for c in categories}
        report['scheme'] = 'category-routed combination' if name == 'current_all30' else 'unified baseline'
        (output / (name + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
        reports.append(report)
    write_comparison(*reports, sources, output / 'comparison.csv')
    (output / 'run.json').write_text(json.dumps(dict(arguments=vars(args), command=command,
                                                    categories=categories, images=len(expected)), indent=2), encoding='utf-8')
    print(f'Complete: {output / "current_all30.json"} and {output / "comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()

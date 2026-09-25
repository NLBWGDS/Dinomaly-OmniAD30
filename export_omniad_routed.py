"""Export fixed category fusion with independent whole-image anomaly scores.

Reads predictions only. Does not read masks, select thresholds or fit weights.
"""

import argparse
import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np


def read_index(folder):
    result = {}
    with (folder / 'scores.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            category = row['category']
            if category in ('.', '..') or '/' in category or '\\' in category:
                raise ValueError('Invalid category')
            parts = PurePosixPath(row['image_path'].replace('\\', '/')).parts
            starts = [i for i in range(len(parts)-1) if parts[i:i+2] == (category, 'test')]
            if len(starts) != 1:
                raise ValueError(f"Ambiguous image path: {row['image_path']}")
            relative = PurePosixPath(*parts[starts[0]+2:])
            if not relative.parts or '..' in relative.parts:
                raise ValueError('Invalid relative image path')
            key = (category, relative.as_posix())
            if key in result:
                raise ValueError(f'Duplicate prediction: {key}')
            nested = folder / category / (relative.as_posix() + '.npy')
            local = folder / category / PurePosixPath(row['map_path'].replace('\\', '/')).name
            path = next((p for p in (nested, local, Path(row['map_path'])) if p.is_file()), nested)
            if not path.is_file():
                raise FileNotFoundError(path)
            result[key] = (row, path)
    return result


def fuse(a, b, weight, ratio):
    if a.ndim != 2 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Maps must have identical 2D shapes and finite values')
    if not np.isfinite(weight) or not 0 <= weight <= 1 or not 0 <= ratio <= 1:
        raise ValueError('Weight and image top ratio must be in [0, 1]')
    flat = a.ravel()
    k = max(1, int(flat.size * ratio))
    image_score = float(np.partition(flat, flat.size-k)[-k:].mean())
    return (1-weight)*a + weight*b, image_score


def proxy(metrics):
    keys = ('pixel_f1', 'pixel_aupro', 'image_f1')
    if any(metrics[k] is None for k in keys):
        return None
    return sum(w * metrics[k] for w, k in zip((25, 6, 18), keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--config', default=str(Path(__file__).with_name('fusion_26_6_18.json')))
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--comparison_report', help='Optional previous comparison JSON for predicted metrics')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['image_source'] != 'baseline_map':
        raise ValueError('Only baseline_map image scoring is supported')
    base, candidate = read_index(Path(args.baseline)), read_index(Path(args.candidate))
    if not base or base.keys() != candidate.keys():
        raise ValueError('Prediction image sets must be identical and nonempty')
    weights = config['pixel_candidate_weights']
    if set(weights) != {k[0] for k in base}:
        raise ValueError('Config categories must exactly match the prediction categories')
    ratio = config['image_top_ratio']
    for weight in weights.values():
        fuse(np.ones((1, 1)), np.ones((1, 1)), weight, ratio)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category', 'image_path', 'score', 'map_path'])
        for (category, relative), (row, path) in base.items():
            a = np.load(path, allow_pickle=False)
            b = np.load(candidate[(category, relative)][1], allow_pickle=False)
            result, image_score = fuse(a, b, weights[category], ratio)
            target = output / category / (relative + '.npy')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, result.astype(np.float32))
            writer.writerow([category, row['image_path'], image_score, str(target.resolve())])
    manifest = dict(config=config, baseline=args.baseline, candidate=args.candidate,
                    note='Image scores are independent of exported pixel maps. Fixed development-selected policy.')
    if args.comparison_report:
        comparison = json.loads(Path(args.comparison_report).read_text(encoding='utf-8'))
        if comparison['arguments']['max_ratio'] != ratio:
            raise ValueError('Comparison uses a different image score ratio')
        estimates = {}
        for category, weight in weights.items():
            name = 'baseline' if weight == 0 else 'candidate' if weight == 1 else 'blend'
            if name == 'blend' and weight != comparison['arguments']['candidate_weight']:
                raise ValueError('Comparison blend weight does not match export')
            previous = comparison['categories'][category]
            metrics = {k: previous[name][k] for k in ('pixel_f1', 'pixel_aupro', 'image_f1')}
            metrics['image_f1'] = previous['baseline']['image_f1']
            metrics['priority_objective'] = proxy(metrics)
            metrics['baseline_priority_objective'] = proxy(previous['baseline'])
            estimates[category] = metrics
            print(category, json.dumps(metrics), flush=True)
        manifest['estimates_from_supplied_report_not_remeasured'] = estimates
    manifest['priority_weights'] = dict(pixel_f1=25, pixel_aupro=6, image_f1=18)
    (output / 'routing_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Exported {len(base)} predictions to {output}')


if __name__ == '__main__':
    main()

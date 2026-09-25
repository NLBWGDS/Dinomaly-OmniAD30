"""Original-coordinate development comparison and optional cached-map fusion."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from diagnose_omniad import locate, read_pair


def f1_hist(positive, negative):
    tp = positive[::-1].cumsum()[::-1]
    fp = negative[::-1].cumsum()[::-1]
    fn = positive.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    return float(f1[i]), float(tp[i] / max(tp[i] + fp[i], 1)), float(tp[i] / max(positive.sum(), 1))


def image_f1(scores, labels):
    if not any(labels) or all(labels):
        return None
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    return float(max(2 * np.count_nonzero((scores >= t) & labels) /
                     max(np.count_nonzero(scores >= t) + labels.sum(), 1)
                     for t in np.unique(scores)))


def region_hist(scores, truth, edges):
    labels, regions = ndimage.label(truth, structure=np.ones((3, 3)))
    if not regions:
        return np.zeros(len(edges) - 1), 0
    sizes = np.bincount(labels.ravel())
    # Every connected defect contributes unit mass, regardless of its area.
    weights = 1.0 / sizes[labels[truth]]
    return np.histogram(scores[truth], bins=edges, weights=weights)[0], regions


def aupro_hist(regional, regions, negative, limit=0.3):
    if not regions or not negative.sum():
        return None
    fpr = np.r_[0., negative[::-1].cumsum() / negative.sum()]
    pro = np.r_[0., regional[::-1].cumsum() / regions]
    # Preserve vertical segments and interpolate exactly to the FPR limit.
    stop = np.searchsorted(fpr, limit, side='right')
    x, y = fpr[:stop], pro[:stop]
    if x[-1] < limit:
        endpoint = np.interp(limit, fpr[stop-1:stop+1], pro[stop-1:stop+1])
        x, y = np.r_[x, limit], np.r_[y, endpoint]
    area = np.sum(np.diff(x) * (y[:-1] + y[1:]) * .5)
    return float(area / limit)


def index_predictions(folder, data, categories):
    result = {}
    with (folder / 'scores.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if categories and row['category'] not in categories:
                continue
            paths = locate(row, folder, data)
            key = str(paths[0].resolve())
            if key in result:
                raise ValueError(f'Duplicate prediction: {key}')
            result[key] = (row['category'], paths)
    return result


def top_score(scores, ratio):
    flat = scores.ravel()
    k = max(1, int(flat.size * ratio))
    return float(np.partition(flat, flat.size - k)[-k:].mean())


def compare(args):
    data = Path(args.data_path)
    selected = args.categories.split(',') if args.categories else None
    base = index_predictions(Path(args.baseline), data, selected)
    candidate = index_predictions(Path(args.candidate), data, selected)
    if not base or base.keys() != candidate.keys():
        raise ValueError('Baseline and candidate must contain exactly the same nonempty image set')
    report = {}
    for category in sorted({v[0] for v in base.values()}):
        keys = [k for k in base if base[k][0] == category]
        low, high = float('inf'), float('-inf')
        for key in keys:
            for source in (base, candidate):
                scores, _ = read_pair(source[key][1])
                low, high = min(low, float(scores.min())), max(high, float(scores.max()))
        edges = np.linspace(low, high if high > low else low + 1e-6, args.bins + 1)
        states = {name: dict(pos=np.zeros(args.bins), neg=np.zeros(args.bins),
                             region=np.zeros(args.bins), regions=0, scores=[], labels=[])
                  for name in ('baseline', 'candidate', 'blend')}
        for key in keys:
            a, truth = read_pair(base[key][1])
            b, other_truth = read_pair(candidate[key][1])
            if not np.array_equal(truth, other_truth):
                raise ValueError(f'Mask mismatch: {key}')
            maps = dict(baseline=a, candidate=b,
                        blend=(1 - args.candidate_weight) * a + args.candidate_weight * b)
            for name, scores in maps.items():
                state = states[name]
                state['pos'] += np.histogram(scores[truth], bins=edges)[0]
                state['neg'] += np.histogram(scores[~truth], bins=edges)[0]
                hist, regions = region_hist(scores, truth, edges)
                state['region'] += hist
                state['regions'] += regions
                state['scores'].append(top_score(scores, args.max_ratio))
                state['labels'].append(base[key][1][2] is not None)
        report[category] = {}
        for name, state in states.items():
            pf1, precision, recall = f1_hist(state['pos'], state['neg'])
            metrics = dict(pixel_f1=pf1, image_f1=image_f1(state['scores'], state['labels']),
                           pixel_aupro=aupro_hist(state['region'], state['regions'], state['neg']),
                           pixel_precision=precision, pixel_recall=recall, images=len(keys))
            report[category][name] = metrics
            print(category, name, json.dumps(metrics), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        protocol='Original resolution; histogram max pixel F1; exact max image F1; '
                 '8-connected region AUPRO integrated to FPR=0.3. Development only.',
        arguments=vars(args), categories=report,
    ), indent=2), encoding='utf-8')
    if args.export_blend_dir:
        destination = Path(args.export_blend_dir)
        if destination.exists():
            raise FileExistsError('Use a new directory for blended predictions')
        destination.mkdir(parents=True)
        with (destination / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['category', 'image_path', 'score', 'map_path'])
            for key, (category, paths) in base.items():
                a = np.load(paths[1], allow_pickle=False)
                b = np.load(candidate[key][1][1], allow_pickle=False)
                blend = (1 - args.candidate_weight) * a + args.candidate_weight * b
                relative = paths[0].relative_to(data / category / 'test')
                target = destination / category / ('__'.join(relative.parts) + '.npy')
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, blend)
                writer.writerow([category, str(paths[0].resolve()), top_score(blend, args.max_ratio),
                                 str(target.resolve())])
        (destination / 'fusion_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--categories')
    parser.add_argument('--candidate_weight', type=float, default=0.5)
    parser.add_argument('--max_ratio', type=float, default=0.01)
    parser.add_argument('--bins', type=int, default=4096)
    parser.add_argument('--output', default='./diagnostics/map_comparison.json')
    parser.add_argument('--export_blend_dir', help='Optional new directory for fixed-weight blended maps')
    args = parser.parse_args()
    if not 0 <= args.candidate_weight <= 1 or not 0 <= args.max_ratio <= 1 or args.bins < 2:
        parser.error('Invalid weight, ratio or bin count')
    compare(args)

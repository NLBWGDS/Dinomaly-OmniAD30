"""Measure exported image scores and maps independently on labeled development data."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_omniad_maps import f1_hist, image_f1, region_hist, aupro_hist
from diagnose_omniad import locate, read_pair
from export_omniad_routed import proxy


def evaluate(predictions, data, bins=4096):
    groups, seen = {}, set()
    with (Path(predictions) / 'scores.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            paths = locate(row, Path(predictions), Path(data))
            key = str(paths[0].resolve())
            score = float(row['score'])
            if key in seen or not np.isfinite(score):
                raise ValueError(f'Duplicate image or nonfinite image score: {key}')
            seen.add(key)
            groups.setdefault(row['category'], []).append((paths, score))
    if not groups:
        raise ValueError('Empty predictions')
    report = {}
    for category, records in sorted(groups.items()):
        low, high = float('inf'), float('-inf')
        for paths, _ in records:
            scores, _ = read_pair(paths)
            low, high = min(low, float(scores.min())), max(high, float(scores.max()))
        edges = np.linspace(low, high if high > low else low + 1e-6, bins + 1)
        pos, neg, regional = (np.zeros(bins) for _ in range(3))
        regions = 0
        image_scores, labels = [], []
        for paths, image_score in records:
            scores, truth = read_pair(paths)
            pos += np.histogram(scores[truth], bins=edges)[0]
            neg += np.histogram(scores[~truth], bins=edges)[0]
            hist, n = region_hist(scores, truth, edges)
            regional += hist
            regions += n
            image_scores.append(image_score)
            labels.append(paths[2] is not None)
        pf1, precision, recall = f1_hist(pos, neg)
        metrics = dict(pixel_f1=pf1 if pos.sum() else None,
                       image_f1=image_f1(image_scores, labels),
                       pixel_aupro=aupro_hist(regional, regions, neg),
                       pixel_precision=precision, pixel_recall=recall,
                       images=len(records), regions=regions,
                       positive_pixels=int(pos.sum()), negative_pixels=int(neg.sum()))
        metrics['priority_objective'] = proxy(metrics)
        report[category] = metrics
        print(category, json.dumps(metrics), flush=True)
    mean = {}
    for metric in ('pixel_f1', 'image_f1', 'pixel_aupro', 'priority_objective'):
        values = [r[metric] for r in report.values()]
        mean[metric] = None if any(v is None for v in values) else float(np.mean(values))
    print('Mean', json.dumps(mean), flush=True)
    return dict(categories=report, mean=mean, image_ids=sorted(seen), bins=bins,
                protocol='Original-resolution histogram max pixel F1 and AUPRO@0.3, '
                         '8-connected regions; max image F1 from scores.csv, not heatmaps. '
                         'Development thresholds, not deployable thresholds. '
                         'Priority weights 25:6:18; not an official competition score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--bins', type=int, default=4096)
    parser.add_argument('--output', default='./diagnostics/export_evaluation.json')
    args = parser.parse_args()
    if args.bins < 2:
        parser.error('--bins must be at least 2')
    report = evaluate(args.predictions, args.data_path, args.bins)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')

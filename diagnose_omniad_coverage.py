"""Development-only attribution of full-coverage regressions; never changes predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_omniad_maps import index_predictions
from diagnose_omniad import read_pair, choose_threshold, panel
from evaluate_omniad_all30 import dataset_inventory


def coverage_regions(shape, side, center_box):
    """Classify original pixels by bilinear support in the resized canvas."""
    height, width = shape
    left, top, right, bottom = center_box
    if not (0 <= left < right <= side and 0 <= top < bottom <= side):
        raise ValueError('Invalid center window')
    x = np.clip((np.arange(width) + .5) * side / width - .5, 0, side-1)
    y = np.clip((np.arange(height) + .5) * side / height - .5, 0, side-1)
    ix = (np.floor(x) >= left) & (np.ceil(x) < right)
    iy = (np.floor(y) >= top) & (np.ceil(y) < bottom)
    ox = (np.ceil(x) < left) | (np.floor(x) >= right)
    oy = (np.ceil(y) < top) | (np.floor(y) >= bottom)
    center = iy[:, None] & ix[None, :]
    border = oy[:, None] | ox[None, :]
    return dict(center=center, border=border, seam=~(center | border))


def regional_counts(scores, truth, regions, threshold):
    predicted = scores >= threshold
    result = {}
    for name, mask in regions.items():
        positive, negative = mask & truth, mask & ~truth
        result[name] = dict(pixels=int(mask.sum()), positives=int(positive.sum()),
                            tp=int((predicted & positive).sum()), fp=int((predicted & negative).sum()),
                            fn=int((~predicted & positive).sum()),
                            positive_score_sum=float(scores[positive].sum(dtype=np.float64)),
                            negative_score_sum=float(scores[negative].sum(dtype=np.float64)))
    return result


def add_counts(total, values):
    for region, values_by_key in values.items():
        destination = total.setdefault(region, {})
        for key, value in values_by_key.items():
            destination[key] = destination.get(key, 0) + value


def finish_counts(regions):
    for values in regions.values():
        values['precision'] = values['tp'] / max(values['tp'] + values['fp'], 1)
        values['recall'] = values['tp'] / max(values['positives'], 1)
        values['f1'] = 2 * values['tp'] / max(2 * values['tp'] + values['fp'] + values['fn'], 1)
        values['positive_score_mean'] = (values['positive_score_sum'] / values['positives']
                                         if values['positives'] else None)
        negatives = values['pixels'] - values['positives']
        values['negative_score_mean'] = values['negative_score_sum'] / negatives if negatives else None


def audit_category(base, candidate, keys, side, center_box, output, bins=4096, top_k=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ranges = {name: [float('inf'), float('-inf')] for name in ('baseline', 'candidate')}
    for key in keys:
        for name, source in [('baseline', base), ('candidate', candidate)]:
            scores, _ = read_pair(source[key][1])
            ranges[name][0] = min(ranges[name][0], float(scores.min()))
            ranges[name][1] = max(ranges[name][1], float(scores.max()))
    edges = {name: np.linspace(low, high if high > low else low + 1e-6, bins+1)
             for name, (low, high) in ranges.items()}
    histograms = {name: [np.zeros(bins, dtype=np.int64), np.zeros(bins, dtype=np.int64)] for name in ranges}
    for key in keys:
        for name, source in [('baseline', base), ('candidate', candidate)]:
            scores, truth = read_pair(source[key][1])
            histograms[name][0] += np.histogram(scores[truth], bins=edges[name])[0]
            histograms[name][1] += np.histogram(scores[~truth], bins=edges[name])[0]
    if any(not h[0].sum() for h in histograms.values()):
        raise ValueError('No positive pixels; max-F1 diagnosis is undefined')
    thresholds = {name: choose_threshold(*histograms[name], edges[name]) for name in ranges}
    totals = {name: {} for name in ('baseline', 'candidate', 'candidate_at_baseline_threshold')}
    rows = []
    center_pixels, difference_sum, difference_max, changed_pixels = 0, 0., 0., 0
    for key in keys:
        a, truth = read_pair(base[key][1])
        b, other_truth = read_pair(candidate[key][1])
        if a.shape != b.shape or not np.array_equal(truth, other_truth):
            raise ValueError(f'Coordinates or masks differ: {key}')
        regions = coverage_regions(a.shape, side, center_box)
        counts = {}
        for name, scores, threshold in [('baseline', a, thresholds['baseline']),
                                         ('candidate', b, thresholds['candidate']),
                                         ('candidate_at_baseline_threshold', b, thresholds['baseline'])]:
            counts[name] = regional_counts(scores, truth, regions, threshold)
            add_counts(totals[name], counts[name])
        differences = np.abs(a[regions['center']].astype(np.float64) - b[regions['center']])
        center_pixels += len(differences)
        difference_sum += float(differences.sum())
        local_max = float(differences.max()) if differences.size else 0.
        difference_max = max(difference_max, local_max)
        changed_pixels += int((differences > 1e-5).sum())
        row = dict(image=key, center_max_abs_change=local_max)
        for region in regions:
            row[region + '_positive_pixels'] = counts['baseline'][region]['positives']
            row[region + '_baseline_fp'] = counts['baseline'][region]['fp']
            row[region + '_candidate_fp'] = counts['candidate'][region]['fp']
            row[region + '_fixed_threshold_fp_delta'] = (counts['candidate_at_baseline_threshold'][region]['fp']
                                                        - counts['baseline'][region]['fp'])
        row['own_threshold_fp_delta'] = sum(v['fp'] for v in counts['candidate'].values()) - sum(v['fp'] for v in counts['baseline'].values())
        rows.append(row)
    for regions in totals.values():
        finish_counts(regions)
    report = dict(images=len(keys), bins=bins, thresholds=thresholds, score_ranges=ranges,
                  region_counts=totals,
                  center_check=dict(pixels=center_pixels, mean_abs_change=difference_sum/max(center_pixels, 1),
                                    max_abs_change=difference_max, changed_above_1e_5=changed_pixels),
                  note='Baseline and candidate each use their category max-F1 threshold. '
                       'candidate_at_baseline_threshold isolates score changes at a fixed decision rule. '
                       'Center invariance does NOT imply unchanged center FP at separately chosen thresholds. '
                       'Seam means bilinear support crosses the crop boundary, not an arbitrary wide band.')
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['own_threshold_fp_delta'], reverse=True))
    low = min(r[0] for r in ranges.values())
    high = max(r[1] for r in ranges.values())
    panels = []
    for rank, row in enumerate(sorted(rows, key=lambda r: r['own_threshold_fp_delta'], reverse=True)[:top_k], 1):
        for name, source in [('baseline', base), ('candidate', candidate)]:
            target = output / f'{rank:02d}_{name}.png'
            panel(source[row['image']][1], thresholds[name], low, high, target)
        panels.append(dict(rank=rank, image=row['image'], own_threshold_fp_delta=row['own_threshold_fp_delta']))
    (output / 'panels.json').write_text(json.dumps(panels, indent=2), encoding='utf-8')
    print(output.name, json.dumps(dict(thresholds=thresholds, center_check=report['center_check'],
        fp_by_region={name: {region: values['fp'] for region, values in regions.items()}
                      for name, regions in totals.items()})), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='./diagnostics/all30_current/current_predictions')
    parser.add_argument('--candidate', default='./predictions/all30_fullcoverage')
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--output_dir', default='./diagnostics/coverage_audit')
    parser.add_argument('--categories', default='ceramic_wafer,chip1,nameplate7,spindle_top')
    parser.add_argument('--bins', type=int, default=4096)
    parser.add_argument('--top_k', type=int, default=3)
    args = parser.parse_args()
    if args.bins < 2 or args.top_k < 0:
        parser.error('bins >= 2 and top_k >= 0 are required')
    selected = {c.strip() for c in args.categories.split(',') if c.strip()}
    manifest = json.loads((Path(args.candidate) / 'coverage_manifest.json').read_text(encoding='utf-8'))
    if not selected or not selected <= set(manifest['selected_categories']):
        raise ValueError('Only categories changed by the coverage experiment can be audited')
    categories, expected = dataset_inventory(args.data_path)
    base = index_predictions(Path(args.baseline), Path(args.data_path), None)
    candidate = index_predictions(Path(args.candidate), Path(args.data_path), None)
    expected_keys = {str(p.resolve()) for p in expected.values()}
    if set(base) != expected_keys or set(candidate) != expected_keys:
        raise ValueError('Both exports must match the complete all30 dataset image set')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for category in sorted(selected):
        keys = sorted(k for k in base if base[k][0] == category)
        summaries[category] = audit_category(base, candidate, keys, manifest['image_size'],
                                              tuple(manifest['windows'][0]), output/category,
                                              args.bins, args.top_k)
    (output / 'summary.json').write_text(json.dumps(dict(categories=summaries,
        all30_inventory_images=len(expected_keys), note='Diagnostic only. No maps or model weights changed; '
        'the unselected 26 categories are not re-scored here.'), indent=2), encoding='utf-8')
    print(f'Diagnostic report: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()

"""Normal-variability-gated reblend of cached detail predictions, without GPU inference.

Reads no masks and fits no parameters to evaluation images. Reliability is a
normal-spread heuristic, not a probability of correctness on anomalous images.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_omniad_all30 import dataset_inventory, require_exact_keys
from export_omniad_routed import read_index
from omniad_normal_calibration import resize_grid
from predict_omniad_fullcoverage import export_predictions


def normal_reliability(calibration, floor=.25):
    if not np.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('Reliability floor must be in [0,1]')
    median = np.asarray(calibration['median'], dtype=np.float32)
    high = np.asarray(calibration['high'], dtype=np.float32)
    ref_span = float(calibration['reference_high']) - float(calibration['reference_median'])
    if (median.ndim != 2 or min(median.shape) < 2 or median.shape != high.shape
            or not np.isfinite(median).all() or not np.isfinite(high).all()
            or np.any(high < median) or not np.isfinite(ref_span) or ref_span <= 1e-6
            or int(calibration['normal_images']) < 5):
        raise ValueError('Invalid normal calibration statistics')
    # High normal variability reduces trust in detail corrections at this location.
    return np.clip(ref_span / np.maximum(high - median, ref_span), floor, 1.).astype(np.float32)


def gated_reblend(baseline, blended, reliability, down_weight=.5):
    if (baseline.ndim != 2 or baseline.shape != blended.shape
            or not np.isfinite(baseline).all() or not np.isfinite(blended).all()
            or np.any(baseline < 0) or np.any(blended < 0)):
        raise ValueError('Expected finite nonnegative same-sized 2D maps')
    if not np.isfinite(down_weight) or not 0 <= down_weight <= 1:
        raise ValueError('down_weight must be in [0,1]')
    if (reliability.ndim != 2 or min(reliability.shape) < 1
            or not np.isfinite(reliability).all()
            or np.any(reliability < 0) or np.any(reliability > 1)):
        raise ValueError('Reliability must be a finite 2D grid in [0,1]')
    gate = resize_grid(reliability, baseline.shape, 'bilinear')
    # blended - baseline already contains the original detail mixture weight.
    # This avoids amplifying float32 error by reconstructing the detail branch.
    gate *= np.where(blended < baseline, down_weight, 1.)
    result = baseline + gate * (blended - baseline)
    return np.clip(result, np.minimum(baseline, blended), np.maximum(baseline, blended)).astype(np.float32)


def validate_source(manifest, baseline, selected):
    args = manifest.get('arguments', {})
    if (args.get('detail_scale') != 2 or not args.get('normal_calibration')
            or manifest.get('labels_used') is not False):
        raise ValueError('Source must be a normal-calibrated detail-scale-2 export')
    weight = args.get('detail_weight', 0)
    if not np.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError('Invalid original detail weight')
    if Path(args.get('predictions', '')).resolve() != Path(baseline).resolve():
        raise ValueError('Baseline must be the ORIGINAL source of the detail export, not its blended output')
    if not selected or not selected <= set(manifest.get('selected_categories', [])):
        raise ValueError('Requested categories must have cached detail predictions')


def run(args):
    if not np.isfinite(args.down_weight) or not 0 <= args.down_weight <= 1:
        raise ValueError('down_weight must be in [0,1]')
    source = Path(args.detail_predictions)
    manifest = json.loads((source / 'coverage_manifest.json').read_text(encoding='utf-8'))
    selected = {c.strip() for c in args.categories.split(',') if c.strip()}
    validate_source(manifest, args.baseline, selected)
    _, expected = dataset_inventory(args.data_path)
    baseline, detail = read_index(Path(args.baseline)), read_index(source)
    require_exact_keys(baseline, expected, 'Original detail baseline')
    require_exact_keys(detail, expected, 'Cached detail predictions')
    if not selected <= {key[0] for key in expected}:
        raise ValueError('Requested categories missing from dataset')
    for key in expected:
        if float(baseline[key][0]['score']) != float(detail[key][0]['score']):
            raise ValueError(f'Image scores differ between sources: {key}')
    reliability = {}
    for category in sorted(selected):
        with np.load(source / f'{category}_detail_calibration.npz', allow_pickle=False) as stats:
            reliability[category] = normal_reliability(stats, args.reliability_floor)
    key_by_path = {str(path.resolve()): key for key, path in expected.items()}
    totals = {category: dict(pixels=0, raised=0, lowered=0, absolute_change=0.) for category in selected}

    def predict(path):
        key = key_by_path[str(path.resolve())]
        a = np.load(baseline[key][1], allow_pickle=False)
        b = np.load(detail[key][1], allow_pickle=False)
        output = gated_reblend(a, b, reliability[key[0]], args.down_weight)
        difference = output - b
        stats = totals[key[0]]
        stats['pixels'] += output.size
        stats['raised'] += int(np.count_nonzero(difference > 1e-7))
        stats['lowered'] += int(np.count_nonzero(difference < -1e-7))
        stats['absolute_change'] += float(np.abs(difference).sum(dtype=np.float64))
        return output

    # Unselected categories come from the accepted detail candidate, not an older baseline.
    export_predictions(detail, expected, selected, args.output_dir, predict)
    report = dict(arguments=vars(args), baseline=str(Path(args.baseline).resolve()),
                  detail_predictions=str(source.resolve()), labels_used=False, model_inference=False,
                  selected_categories=sorted(selected), changes_vs_detail=totals,
                  image_scores='Copied unchanged from detail export',
                  note='Fixed heuristic ablation. No guaranteed precision/recall gain. '
                       'Keep source directories; unselected maps are referenced without copying.')
    for category in sorted(selected):
        stats = totals[category]
        print(category, json.dumps(dict(mean_reliability=float(reliability[category].mean()),
              mean_abs_change=stats['absolute_change']/stats['pixels'],
              raised_fraction=stats['raised']/stats['pixels'], lowered_fraction=stats['lowered']/stats['pixels'])))
    (Path(args.output_dir) / 'reblend_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--detail_predictions', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--categories', default='ceramic_wafer,spindle_top')
    parser.add_argument('--reliability_floor', type=float, default=.25)
    parser.add_argument('--down_weight', type=float, default=.5)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

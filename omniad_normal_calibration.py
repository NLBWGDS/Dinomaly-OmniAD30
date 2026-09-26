"""Normal-only spatial score calibration for newly predicted legacy borders."""

import numpy as np
import torch
import torch.nn.functional as F


def resize_grid(array, shape, mode):
    tensor = torch.as_tensor(np.asarray(array), dtype=torch.float32)[None, None]
    options = {} if mode == 'area' else {'align_corners': False}
    return F.interpolate(tensor, size=shape, mode=mode, **options)[0, 0].numpy()


def fit_normal_calibration(normal_maps, side, center_box, grid=32):
    """Fit per-location median/95th percentile from normal training maps only.

    Only small area-averaged grids are retained, bounding memory as image count
    grows. The central normal score distribution defines the reference scale.
    """
    if not 2 <= grid <= side:
        raise ValueError('Require 2 <= calibration grid <= canvas side')
    left, top, right, bottom = center_box
    if not (0 <= left < right <= side and 0 <= top < bottom <= side):
        raise ValueError('Invalid center box')
    samples = []
    for scores in normal_maps:
        if scores.shape != (side, side) or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError('Normal calibration maps must be finite, nonnegative and canvas-sized')
        samples.append(resize_grid(scores, (grid, grid), 'area'))
    if len(samples) < 5:
        raise ValueError('At least five train/good images are required for spatial calibration')
    samples = np.stack(samples)
    # Reference cells must lie wholly within the original center view.
    starts = np.arange(grid) * side / grid
    ends = (np.arange(grid) + 1) * side / grid
    inside = ((starts >= top) & (ends <= bottom))[:, None] & ((starts >= left) & (ends <= right))[None, :]
    if not inside.any():
        raise ValueError('No full calibration cells inside center crop')
    reference = samples[:, inside]
    reference_median, reference_high = np.quantile(reference, [.5, .95])
    if reference_high - reference_median <= 1e-6:
        raise ValueError('Degenerate normal reference spread; refusing arbitrary calibration')
    median, high = np.quantile(samples, [.5, .95], axis=0)
    return dict(median=median.astype(np.float32), high=high.astype(np.float32),
                reference_median=float(reference_median), reference_high=float(reference_high),
                normal_images=len(samples), side=side, center_box=np.array(center_box, dtype=np.int64))


def calibrate_border(scores, calibration, strength=.5):
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Calibration strength must be in [0,1]')
    side = int(calibration['side'])
    if scores.shape != (side, side) or not np.isfinite(scores).all():
        raise ValueError('Invalid calibration input map')
    if strength == 0:
        return scores.copy()
    median = resize_grid(calibration['median'], scores.shape, 'bilinear')
    high = resize_grid(calibration['high'], scores.shape, 'bilinear')
    ref_median = float(calibration['reference_median'])
    ref_span = float(calibration['reference_high']) - ref_median
    if not np.isfinite(ref_span) or ref_span <= 1e-6:
        raise ValueError('Invalid normal reference spread')
    # Bound amplification in near-constant normal regions and attenuation in
    # variable regions. The bounds are fixed, not fitted to development labels.
    gain = np.clip(ref_span / np.maximum(high - median, ref_span * .25), .25, 4.)
    calibrated = np.maximum((scores - median) * gain + ref_median, 0)
    result = ((1-strength)*scores + strength*calibrated).astype(np.float32)
    left, top, right, bottom = map(int, calibration['center_box'])
    result[top:bottom, left:right] = scores[top:bottom, left:right]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite calibrated map')
    return result

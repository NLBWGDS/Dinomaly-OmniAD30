"""Normal patch retrieval helpers; no labels or anomaly masks are consumed."""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def coreset_indices(features, count, projection_dim=64, seed=1):
    """Projected farthest-first selection, without a quadratic distance matrix.

    Inspired by PatchCore coreset coverage. Returns original row indices so
    source-image ownership survives selection for leave-one-image-out fitting.
    Projection and initialization differ from the official implementation.
    """
    if features.ndim != 2 or min(features.shape) < 1 or not 1 <= count <= len(features):
        raise ValueError('Invalid candidate matrix or coreset count')
    if projection_dim < 1 or not torch.isfinite(features).all():
        raise ValueError('Projection dimension must be positive and candidates finite')
    if count == len(features):
        return torch.arange(count, device=features.device)
    generator = torch.Generator().manual_seed(seed)
    x = features.float()
    if x.shape[1] > projection_dim:
        projection = torch.randn(x.shape[1], projection_dim, generator=generator) / math.sqrt(projection_dim)
        x = x @ projection.to(x.device)
    norms = x.square().sum(dim=1)
    closest = torch.full((len(x),), float('inf'), device=x.device)
    current = torch.randint(len(x), (1,), generator=generator).to(x.device).squeeze(0)
    selected = torch.empty(count, dtype=torch.long, device=x.device)
    for i in range(count):
        selected[i] = current
        distance = (norms + norms[current] - 2 * (x @ x[current])).clamp_min(0)
        closest = torch.minimum(closest, distance)
        # Negative sentinels prevent duplicates even when all candidates coincide.
        closest[current] = -1
        current = closest.argmax()
    return selected


def descriptors(features, weights):
    if not features or len(features) != len(weights):
        raise ValueError('One weight per feature group is required')
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Feature weights must be finite, nonnegative and have a positive sum')
    shape = features[0].shape
    if len(shape) != 4 or any(x.ndim != 4 or x.shape[0] != shape[0] or x.shape[2:] != shape[2:]
                              for x in features):
        raise ValueError('Feature groups must share batch and spatial dimensions')
    normalized = [F.normalize(x.float(), dim=1) * math.sqrt(w / sum(weights))
                  for x, w in zip(features, weights)]
    return torch.cat(normalized, dim=1).permute(0, 2, 3, 1).reshape(-1, sum(x.shape[1] for x in features))


def valid_patches(height, width, grid_h, grid_w, input_size):
    """Select full patches inside the real letterboxed image, excluding padding."""
    scale = min(input_size / width, input_size / height)
    rh, rw = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (input_size - rh) // 2, (input_size - rw) // 2
    y0 = torch.arange(grid_h, dtype=torch.float64) * input_size / grid_h
    x0 = torch.arange(grid_w, dtype=torch.float64) * input_size / grid_w
    ys = (y0 >= top) & (y0 + input_size / grid_h <= top + rh)
    xs = (x0 >= left) & (x0 + input_size / grid_w <= left + rw)
    mask = (ys[:, None] & xs[None, :]).flatten()
    if not mask.any():
        raise ValueError('Image is too narrow for a complete non-padding patch')
    return mask


def nearest_distance(query, bank, chunk=256, owners=None, exclude_owner=None):
    """Exact cosine nearest-neighbor distance with bounded similarity memory."""
    if chunk < 1 or query.ndim != 2 or bank.ndim != 2 or not bank.shape[0] or query.shape[1] != bank.shape[1]:
        raise ValueError('Invalid query/bank dimensions or chunk size')
    if not torch.isfinite(query).all() or not torch.isfinite(bank).all():
        raise ValueError('Descriptors must be finite')
    if exclude_owner is not None:
        if owners is None or owners.shape != (bank.shape[0],):
            raise ValueError('Bank owners required for leave-one-image-out calibration')
        keep = owners != exclude_owner
        if not keep.any():
            raise ValueError('No other normal images available for calibration')
        bank = bank[keep]
    scores = []
    for part in query.split(chunk):
        scores.append((1 - (part @ bank.T).amax(dim=1)).clamp(0, 2))
    return torch.cat(scores)


def calibration_scale(normal_reconstruction, normal_memory):
    """Match normal-only 95th percentiles, never test-image extrema."""
    a, b = normal_reconstruction.float(), normal_memory.float()
    if not a.numel() or not b.numel() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Calibration requires nonempty finite normal scores')
    aq, bq = torch.quantile(a, .95).item(), torch.quantile(b, .95).item()
    if aq <= 1e-8 or bq <= 1e-8:
        raise ValueError('Degenerate normal calibration; do not export an arbitrary score scale')
    return aq / bq

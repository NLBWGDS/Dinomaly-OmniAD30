"""Normal-only reconstruction objectives for local anomaly maps."""

import math

import torch
import torch.nn.functional as F


def local_hard_loss(encoder_features, decoder_features, hard_fraction=0.1):
    """Balance average error and the hardest normal locations in each image.

    Mining separately per image prevents one difficult product from consuming
    the entire batch's gradient budget. No anomaly masks are used.
    """
    if not 0 < hard_fraction <= 1:
        raise ValueError("hard_fraction must be in (0, 1]")
    if not encoder_features or len(encoder_features) != len(decoder_features):
        raise ValueError("Feature lists must be nonempty and equally sized")
    losses = []
    for target, prediction in zip(encoder_features, decoder_features):
        error = (1 - F.cosine_similarity(target.detach(), prediction, dim=1)).clamp_min(0)
        error = error.flatten(1)
        count = max(1, math.ceil(error.shape[1] * hard_fraction))
        hard = error.topk(count, dim=1).values.mean(dim=1)
        losses.append((0.5 * error.mean(dim=1) + 0.5 * hard).mean())
    return torch.stack(losses).mean()

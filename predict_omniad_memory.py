"""Experimental normal-reference branch on an existing Dinomaly checkpoint.

Uses train/good only to build/calibrate a bank; image scores remain unchanged.
Inspired by normal patch retrieval in PatchCore, not a PatchCore reproduction.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from export_omniad_routed import read_index
from omniad_memory import descriptors, valid_patches, nearest_distance, calibration_scale, coreset_indices


def run(args):
    from dinomaly_omniad_uni import (load_checkpoint_model, apply_checkpoint_preprocessing,
                                    get_omniad_transforms, restore_anomaly_map, setup_seed)

    weights = [float(x) for x in args.feature_weights.split(',')]
    if len(weights) != 2 or not all(np.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Use two finite nonnegative feature weights with positive sum')
    if not np.isfinite(args.memory_weight) or not 0 <= args.memory_weight <= 1:
        raise ValueError('memory_weight must be in [0,1]')
    if args.bank_size < 2 or args.query_chunk < 1:
        raise ValueError('bank_size >= 2 and query_chunk >= 1 are required')
    sampling = getattr(args, 'sampling', 'random')
    multiplier = getattr(args, 'candidate_multiplier', 4)
    projection_dim = getattr(args, 'projection_dim', 64)
    if sampling not in ('random', 'coreset') or multiplier < 1 or projection_dim < 1:
        raise ValueError('Invalid sampling mode, candidate multiplier or projection dimension')
    records = read_index(Path(args.predictions))
    if not records or args.category not in {key[0] for key in records}:
        raise ValueError('Selected category is absent from source predictions')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new output directory; baseline must be preserved')
    data = Path(args.data_path)
    for (category, relative), (row, _) in records.items():
        if not np.isfinite(float(row['score'])):
            raise ValueError('Nonfinite source image score')
        if category == args.category and not (data / category / 'test' / relative).is_file():
            raise FileNotFoundError(data / category / 'test' / relative)
    normal_root = data / args.category / 'train' / 'good'
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    normal_paths = sorted(p for p in normal_root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    if not 2 <= len(normal_paths) <= args.bank_size:
        raise ValueError('Need at least two normal images and bank_size >= normal image count')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu only for debugging')
    setup_seed(args.seed)
    model_args = SimpleNamespace(checkpoint=args.checkpoint, encoder='dinov2reg_vit_base_14',
                                 image_size=560, crop_size=560)
    model, checkpoint = load_checkpoint_model(model_args, device)
    if checkpoint.get('categories') and args.category not in checkpoint['categories']:
        raise ValueError('Selected category was not included in this checkpoint training')
    apply_checkpoint_preprocessing(model_args, checkpoint)
    if model_args.preprocess != 'letterbox':
        raise ValueError('Normal reference inference currently requires a letterbox checkpoint')
    transform, _ = get_omniad_transforms(model_args)

    def extract(path):
        with Image.open(path) as image:
            image = image.convert('RGB')
            width, height = image.size
            batch = transform(image).unsqueeze(0).to(device)
        en, de = model(batch)
        h, w = en[0].shape[-2:]
        valid = valid_patches(height, width, h, w, model_args.crop_size).to(device)
        patches = descriptors(en, weights)
        reconstruction = sum(weight / sum(weights) * (1 - F.cosine_similarity(a, b, dim=1))
                             for a, b, weight in zip(en, de, weights)).clamp_min(0).flatten()
        return patches, reconstruction, valid, (height, width, h, w)

    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(args.seed)
    quota = args.bank_size // len(normal_paths)
    candidate_quota = quota * (multiplier if sampling == 'coreset' else 1)
    with torch.inference_mode():
        chunks, owner_chunks = [], []
        for i, path in enumerate(normal_paths):
            patches, _, valid, _ = extract(path)
            patches = patches[valid].cpu()
            indices = torch.randperm(len(patches), generator=generator)[:candidate_quota]
            chunks.append(patches[indices])
            owner_chunks.append(torch.full((len(indices),), i, dtype=torch.long))
            print(f'Normal bank: {i+1}/{len(normal_paths)}', flush=True)
        bank = torch.cat(chunks).to(device)
        owners = torch.cat(owner_chunks).to(device)
        del chunks, owner_chunks
        candidate_count = len(bank)
        selection_start = time.perf_counter()
        if sampling == 'coreset':
            target_count = min(quota * len(normal_paths), len(bank))
            print(f'Coreset selection: {len(bank)} candidates -> {target_count} patches', flush=True)
            indices = coreset_indices(bank, target_count, projection_dim, args.seed)
            bank, owners = bank[indices], owners[indices]
            if owners.unique().numel() < 2:
                raise ValueError('Coreset covers fewer than two images; increase bank_size')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        selection_seconds = time.perf_counter() - selection_start
        normal_rec, normal_mem = [], []
        for i, path in enumerate(normal_paths):
            patches, reconstruction, valid, _ = extract(path)
            # Exclude ALL patches of this image, not merely an identical patch.
            distances = nearest_distance(patches[valid], bank, args.query_chunk, owners, i)
            normal_rec.append(reconstruction[valid].cpu())
            normal_mem.append(distances.cpu())
            print(f'Normal calibration: {i+1}/{len(normal_paths)}', flush=True)
        scale = calibration_scale(torch.cat(normal_rec), torch.cat(normal_mem))
        torch.save(dict(bank=bank.cpu(), owners=owners.cpu(), scale=scale,
                        normal_paths=[str(p.resolve()) for p in normal_paths],
                        arguments=vars(args), crop_size=model_args.crop_size,
                        sampling=sampling, candidate_count=candidate_count), output / 'normal_bank.pt')
        print(f'Bank patches: {len(bank)}, normal-only scale: {scale:.6f}', flush=True)
        timings = []
        with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['category', 'image_path', 'score', 'map_path'])
            for (category, relative), (row, path) in records.items():
                target = output / category / (relative + '.npy')
                target.parent.mkdir(parents=True, exist_ok=True)
                if category == args.category:
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    patches, _, _, (height, width, h, w) = extract(data / category / 'test' / relative)
                    distance = nearest_distance(patches, bank, args.query_chunk).reshape(1, 1, h, w)
                    distance = F.interpolate(distance, size=(model_args.crop_size, model_args.crop_size),
                                             mode='bilinear', align_corners=False)
                    memory = restore_anomaly_map(distance, height, width, model_args)[0, 0].cpu().numpy() * scale
                    baseline = np.load(path, allow_pickle=False)
                    if baseline.shape != memory.shape or not np.isfinite(baseline).all():
                        raise ValueError(f'Invalid original-resolution source map: {path}')
                    fused = (1 - args.memory_weight) * baseline + args.memory_weight * memory
                    np.save(target, fused.astype(np.float32))
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    timings.append((time.perf_counter() - start) * 1000)
                    print(f'{category}: {len(timings)} maps exported', flush=True)
                else:
                    shutil.copyfile(path, target)
                writer.writerow([category, row['image_path'], row['score'], str(target.resolve())])
    manifest = dict(arguments=vars(args), bank_patches=len(bank), normal_only_scale=scale,
                    normal_images=[str(p.resolve()) for p in normal_paths],
                    image_scores='Copied verbatim; unselected category maps copied byte-for-byte',
                    calibration='95th percentile ratio, leave-one-normal-image-out; no anomaly labels',
                    sampling=sampling, candidate_count=candidate_count,
                    selection_seconds=selection_seconds,
                    sampling_note='Random per-image pool; optional projected farthest-first coreset. '
                                  'Original full-dimensional features retained for retrieval; not full PatchCore.',
                    mean_branch_ms=float(np.mean(timings)),
                    timing_note='Additional branch including IO, no warmup. Excludes prior baseline inference.',
                    status='Experimental; no accuracy or single-model-bonus guarantee')
    (output / 'memory_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Exported {len(records)} predictions to {output}; original image scores preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--data_path', default='../dataset/download/Omni-AD-30-release')
    parser.add_argument('--category', default='infusion_bottle_bottom5')
    parser.add_argument('--feature_weights', default='0.25,0.75')
    parser.add_argument('--memory_weight', type=float, default=0.25)
    parser.add_argument('--bank_size', type=int, default=8192)
    parser.add_argument('--sampling', choices=['random', 'coreset'], default='random')
    parser.add_argument('--candidate_multiplier', type=int, default=4)
    parser.add_argument('--projection_dim', type=int, default=64)
    parser.add_argument('--query_chunk', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())

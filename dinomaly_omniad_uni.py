import argparse
import csv
import logging
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.abspath(os.path.join(REPO_DIR, "..", "dataset", "download", "Omni-AD-30-release"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Letterbox:
    def __init__(self, size, fill, interpolation):
        self.size = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image):
        width, height = image.size
        scale = min(self.size / width, self.size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        image = TF.resize(
            image,
            [resized_height, resized_width],
            interpolation=self.interpolation,
        )
        left = (self.size - resized_width) // 2
        top = (self.size - resized_height) // 2
        right = self.size - resized_width - left
        bottom = self.size - resized_height - top
        return TF.pad(image, [left, top, right, bottom], fill=self.fill)


def get_omniad_transforms(args, train=False):
    if args.preprocess == "legacy":
        from dataset import get_data_transforms

        return get_data_transforms(args.image_size, args.crop_size)

    mean_fill = tuple(round(value * 255) for value in IMAGENET_MEAN)
    image_ops = []
    if train and args.train_augment:
        image_ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05))
    image_ops.append(Letterbox(args.crop_size, mean_fill, InterpolationMode.BICUBIC))
    if train and args.train_augment:
        image_ops.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.02, 0.02),
                interpolation=InterpolationMode.BILINEAR,
                fill=mean_fill,
            )
        )
    image_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    mask_ops = [
        Letterbox(args.crop_size, 0, InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ]
    return transforms.Compose(image_ops), transforms.Compose(mask_ops)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, save_path=None, level="INFO"):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, "log.txt"), encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def discover_categories(data_path, categories=None):
    data_path = os.path.abspath(data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Omni-AD data path does not exist: {data_path}")

    if categories:
        selected = [item.strip() for item in categories.split(",") if item.strip()]
    else:
        selected = sorted(
            item for item in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, item))
        )

    if not selected:
        raise RuntimeError(f"No categories found directly under: {data_path}")

    valid = []
    for item in selected:
        train_good = os.path.join(data_path, item, "train", "good")
        if not os.path.isdir(train_good):
            raise FileNotFoundError(f"{item} is missing train/good: {train_good}")
        valid.append(item)
    return valid


def check_environment(args, item_list, logger):
    logger.info(f"data_path: {os.path.abspath(args.data_path)}")
    logger.info(f"categories: {len(item_list)}")

    total_train = 0
    missing_test = []
    missing_gt = []
    for item in item_list:
        train_good = os.path.join(args.data_path, item, "train", "good")
        train_count = len([
            name for name in os.listdir(train_good)
            if os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ])
        total_train += train_count
        if not os.path.isdir(os.path.join(args.data_path, item, "test")):
            missing_test.append(item)
        if not os.path.isdir(os.path.join(args.data_path, item, "ground_truth")):
            missing_gt.append(item)
        if train_count == 0:
            raise RuntimeError(f"{item} has no images under train/good")

    logger.info(f"train/good images: {total_train}")
    logger.info(f"missing test dirs: {len(missing_test)}")
    logger.info(f"missing ground_truth dirs: {len(missing_gt)}")
    logger.info(f"python: {'.'.join(map(str, os.sys.version_info[:3]))}")
    logger.info(f"torch: {torch.__version__}, cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"cuda device: {torch.cuda.get_device_name(0)}")

    get_omniad_transforms(args)
    logger.info("check passed: dataset layout and core imports are ready.")


def has_dev_labels(data_path, item):
    return (
        os.path.isdir(os.path.join(data_path, item, "test"))
        and os.path.isdir(os.path.join(data_path, item, "ground_truth"))
    )


def build_model(encoder_name, device):
    from dinov1.utils import trunc_normal_
    from models import vit_encoder
    from models.uad import ViTill
    from models.vision_transformer import Block as VitBlock, LinearAttention2, bMlp

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("encoder_name must contain small, base, or large")

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    ).to(device)

    trainable = nn.ModuleList([bottleneck, decoder])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    return model, trainable


def make_train_loader(args, data_transform, item_list):
    train_data_list = []
    for class_idx, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, "train")
        train_data = ImageFolder(root=train_path, transform=data_transform)
        if len(train_data) == 0:
            raise RuntimeError(f"No supported training images found under: {train_path}")
        train_data.classes = [item]
        train_data.class_to_idx = {item: class_idx}
        train_data.samples = [(sample[0], class_idx) for sample in train_data.samples]
        train_data.targets = [class_idx] * len(train_data.samples)
        train_data_list.append(train_data)

    train_data = ConcatDataset(train_data_list)
    sampler = None
    if args.category_balanced and len(train_data_list) > 1:
        sample_weights = []
        for dataset in train_data_list:
            sample_weights.extend([1.0 / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data), replacement=True)

    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    return train_data, loader


def evaluate_dev(model, args, data_transform, gt_transform, item_list, device, logger):
    from dataset import MVTecDataset
    from utils import evaluation_batch

    metrics = []
    for item in item_list:
        if not has_dev_labels(args.data_path, item):
            logger.info(f"{item}: skipped, no development labels found")
            continue

        test_data = MVTecDataset(
            root=os.path.join(args.data_path, item),
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test",
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        result = evaluation_batch(
            model,
            test_loader,
            device,
            max_ratio=args.max_ratio,
            resize_mask=args.eval_mask_size,
            gaussian_kernel_size=args.gaussian_kernel_size,
            gaussian_sigma=args.gaussian_sigma,
            feature_weights=get_feature_weights(args, item),
        )
        metrics.append(result)
        logger.info(
            "{}: I-AUROC:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(
                item, *result
            )
        )

    if metrics:
        mean_result = np.mean(np.array(metrics), axis=0)
        logger.info(
            "Mean: I-AUROC:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(
                *mean_result
            )
        )
    return metrics


def train(args, item_list, device, logger):
    from optimizers import StableAdamW
    from utils import WarmCosineScheduler, global_cosine_hm_percent

    setup_seed(args.seed)
    train_transform, _ = get_omniad_transforms(args, train=True)
    data_transform, gt_transform = get_omniad_transforms(args)
    train_data, train_loader = make_train_loader(args, train_transform, item_list)
    model, trainable = build_model(args.encoder, device)

    optimizer = StableAdamW(
        [{"params": trainable.parameters()}],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer,
        base_value=args.lr,
        final_value=args.final_lr,
        total_iters=args.total_iters,
        warmup_iters=args.warmup_iters,
    )

    logger.info(f"categories: {', '.join(item_list)}")
    logger.info(f"train images: {len(train_data)}")
    logger.info(f"encoder: {args.encoder}")
    logger.info(
        f"preprocess: {args.preprocess}, input: {args.crop_size}, "
        f"category_balanced: {args.category_balanced}, augmentation: {args.train_augment}"
    )
    logger.info("compliance: training reads Omni-AD train/good only; dev labels are used only for optional evaluation.")

    it = 0
    while it < args.total_iters:
        model.train()
        loss_list = []
        for img, _ in train_loader:
            img = img.to(device, non_blocking=True)
            en, de = model(img)

            p = min(args.hm_percent * it / args.hm_warmup_iters, args.hm_percent)
            loss = global_cosine_hm_percent(en, de, p=p, factor=args.hm_factor)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(loss.item())
            it += 1

            if it % args.log_every == 0:
                logger.info(f"iter [{it}/{args.total_iters}], loss:{np.mean(loss_list):.4f}")
                loss_list = []

            if args.eval_every > 0 and it % args.eval_every == 0:
                evaluate_dev(model, args, data_transform, gt_transform, item_list, device, logger)
                model.train()

            if it >= args.total_iters:
                break

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "omniad_dinomaly_uni.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "categories": item_list,
            "encoder": args.encoder,
            "image_size": args.image_size,
            "crop_size": args.crop_size,
            "eval_mask_size": args.eval_mask_size,
            "preprocess": args.preprocess,
            "gaussian_kernel_size": args.gaussian_kernel_size,
            "gaussian_sigma": args.gaussian_sigma,
        },
        checkpoint_path,
    )
    logger.info(f"saved checkpoint: {checkpoint_path}")
    return checkpoint_path


class PredictDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.paths = []
        for current_root, _, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    self.paths.append(os.path.join(current_root, name))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        original_size = image.size[::-1]
        return self.transform(image), path, torch.tensor(original_size, dtype=torch.int64)


def load_checkpoint_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder_name = checkpoint.get("encoder", args.encoder)
    model, _ = build_model(encoder_name, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def apply_checkpoint_preprocessing(args, checkpoint):
    args.image_size = checkpoint.get("image_size", args.image_size)
    args.crop_size = checkpoint.get("crop_size", args.crop_size)
    # Checkpoints created before letterbox support used resize + center crop.
    args.preprocess = checkpoint.get("preprocess", "legacy")
    args.eval_mask_size = checkpoint.get(
        "eval_mask_size",
        256 if "preprocess" not in checkpoint else args.crop_size,
    )


def restore_anomaly_map(anomaly_map, original_h, original_w, args):
    if args.preprocess == "letterbox":
        target = anomaly_map.shape[-1]
        scale = min(target / original_w, target / original_h)
        resized_w = max(1, round(original_w * scale))
        resized_h = max(1, round(original_h * scale))
        left = (target - resized_w) // 2
        top = (target - resized_h) // 2
        anomaly_map = anomaly_map[:, :, top:top + resized_h, left:left + resized_w]
    else:
        canvas = anomaly_map.new_zeros((1, 1, args.image_size, args.image_size))
        offset = max(0, (args.image_size - args.crop_size) // 2)
        crop = F.interpolate(
            anomaly_map,
            size=(args.crop_size, args.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        canvas[:, :, offset:offset + args.crop_size, offset:offset + args.crop_size] = crop
        anomaly_map = canvas

    return F.interpolate(
        anomaly_map,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )


def score_anomaly_map(anomaly_map, original_h, original_w, args):
    if args.preprocess == "letterbox":
        target = anomaly_map.shape[-1]
        scale = min(target / original_w, target / original_h)
        resized_w = max(1, round(original_w * scale))
        resized_h = max(1, round(original_h * scale))
        left = (target - resized_w) // 2
        top = (target - resized_h) // 2
        anomaly_map = anomaly_map[:, :, top:top + resized_h, left:left + resized_w]

    flat_map = anomaly_map.flatten()
    topk = max(1, int(flat_map.numel() * args.max_ratio))
    return torch.topk(flat_map, topk).values.mean().item()


def predict(args, item_list, device, logger):
    from utils import cal_anomaly_maps, get_gaussian_kernel

    model, checkpoint = load_checkpoint_model(args, device)
    apply_checkpoint_preprocessing(args, checkpoint)
    data_transform, _ = get_omniad_transforms(args)
    gaussian_kernel = get_gaussian_kernel(
        kernel_size=args.gaussian_kernel_size,
        sigma=args.gaussian_sigma,
    ).to(device)
    os.makedirs(args.output_dir, exist_ok=True)

    score_path = os.path.join(args.output_dir, "scores.csv")
    with open(score_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "image_path", "score", "map_path"])

        with torch.no_grad():
            for item in item_list:
                test_root = os.path.join(args.data_path, item, "test")
                if not os.path.isdir(test_root):
                    logger.info(f"{item}: skipped, no test directory found")
                    continue
                dataset = PredictDataset(test_root, data_transform)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                item_output = os.path.join(args.output_dir, item)
                os.makedirs(item_output, exist_ok=True)

                for img, paths, original_sizes in loader:
                    img = img.to(device, non_blocking=True)
                    en, de = model(img)
                    anomaly_map, _ = cal_anomaly_maps(
                        en,
                        de,
                        img.shape[-1],
                        feature_weights=get_feature_weights(args, item),
                    )
                    anomaly_map = gaussian_kernel(anomaly_map)

                    for batch_idx, path in enumerate(paths):
                        original_h = int(original_sizes[batch_idx][0])
                        original_w = int(original_sizes[batch_idx][1])
                        restored_map = restore_anomaly_map(
                            anomaly_map[batch_idx:batch_idx + 1],
                            original_h,
                            original_w,
                            args,
                        )
                        image_score = score_anomaly_map(
                            anomaly_map[batch_idx:batch_idx + 1],
                            original_h,
                            original_w,
                            args,
                        )
                        resized_map = restored_map[0, 0].cpu().numpy()
                        rel = os.path.relpath(path, test_root)
                        safe_name = rel.replace(os.sep, "__")
                        map_path = os.path.join(item_output, os.path.splitext(safe_name)[0] + ".npy")
                        np.save(map_path, resized_map)
                        writer.writerow([item, path, image_score, map_path])

    logger.info(f"saved prediction scores: {score_path}")
    logger.info(f"checkpoint categories: {', '.join(checkpoint.get('categories', []))}")


def parse_args():
    parser = argparse.ArgumentParser(description="Dinomaly reproduction entry for Omni-AD style data.")
    parser.add_argument("--mode", choices=["check", "train", "eval", "predict"], default="train")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category list. Default: auto-discover.")
    parser.add_argument("--output_dir", type=str, default="./saved_results/omniad_dinomaly_uni")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--image_size", type=int, default=560, help="Legacy resize size; retained for old checkpoints.")
    parser.add_argument("--crop_size", type=int, default=560, help="Model input size. Use a multiple of the patch size.")
    parser.add_argument("--eval_mask_size", type=int, default=560)
    parser.add_argument("--preprocess", choices=["letterbox", "legacy"], default="letterbox")
    parser.add_argument("--train_augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--category_balanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaussian_kernel_size", type=int, default=3)
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument(
        "--feature_weights",
        type=parse_feature_weights,
        default=None,
        help="Comma-separated shallow,deep anomaly-map weights. Default: 0.5,0.5.",
    )
    parser.add_argument(
        "--category_feature_weights",
        type=parse_category_feature_weights,
        default={},
        help="Per-category overrides, e.g. 'wafer2=0.25,0.75;iron_lattice=0.5,0.5'.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--total_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--final_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--hm_percent", type=float, default=0.9)
    parser.add_argument("--hm_warmup_iters", type=int, default=1000)
    parser.add_argument("--hm_factor", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--max_ratio", type=float, default=0.01)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=0, help="Set above 0 only for a labeled development split.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_feature_weights(value):
    try:
        weights = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("feature weights must be comma-separated numbers") from exc
    if len(weights) != 2:
        raise argparse.ArgumentTypeError("exactly two feature weights are required: shallow,deep")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise argparse.ArgumentTypeError("feature weights must be non-negative with a positive sum")
    return weights


def parse_category_feature_weights(value):
    result = {}
    for assignment in value.split(";"):
        assignment = assignment.strip()
        if not assignment:
            continue
        if "=" not in assignment:
            raise argparse.ArgumentTypeError(
                "category feature weights must use category=shallow,deep"
            )
        category, weights = assignment.split("=", 1)
        category = category.strip()
        if not category or category in result:
            raise argparse.ArgumentTypeError("category names must be non-empty and unique")
        result[category] = parse_feature_weights(weights)
    return result


def get_feature_weights(args, category):
    return args.category_feature_weights.get(category, args.feature_weights)


def validate_args(args):
    if args.crop_size <= 0 or args.eval_mask_size <= 0:
        raise ValueError("--crop_size and --eval_mask_size must be positive")
    if args.gaussian_kernel_size <= 0 or args.gaussian_kernel_size % 2 == 0:
        raise ValueError("--gaussian_kernel_size must be a positive odd number")
    if not 0 <= args.max_ratio <= 1:
        raise ValueError("--max_ratio must be between 0 and 1")
    if args.preprocess == "legacy" and args.image_size < args.crop_size:
        raise ValueError("legacy preprocessing requires --image_size >= --crop_size")
    unknown_categories = set(args.category_feature_weights) - set(discover_categories(args.data_path, args.categories))
    if unknown_categories:
        raise ValueError(f"Unknown categories in --category_feature_weights: {sorted(unknown_categories)}")
    patch_size = int(args.encoder.rsplit("_", 1)[-1])
    if args.crop_size % patch_size != 0:
        raise ValueError(f"--crop_size must be divisible by encoder patch size {patch_size}")


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    logger = get_logger("omniad_dinomaly_uni", None if args.mode == "check" else args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"device: {device}")

    item_list = discover_categories(args.data_path, args.categories)
    if args.mode == "check":
        check_environment(args, item_list, logger)
    elif args.mode == "train":
        train(args, item_list, device, logger)
    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for eval mode")
        model, checkpoint = load_checkpoint_model(args, device)
        apply_checkpoint_preprocessing(args, checkpoint)
        data_transform, gt_transform = get_omniad_transforms(args)
        evaluate_dev(model, args, data_transform, gt_transform, item_list, device, logger)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for predict mode")
        predict(args, item_list, device, logger)

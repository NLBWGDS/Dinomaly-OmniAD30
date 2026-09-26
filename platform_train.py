"""Competition training entry: /input normal images -> /output/output.bin."""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from platform_io import bounded_number, discover_images, get_parameter, load_json, parse_bool


ITERATION = re.compile(r'iter \[(\d+)/(\d+)\], loss:([0-9.eE+-]+)')


class StateLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('w', encoding='utf-8', buffering=1)

    def write(self, message):
        print(message, flush=True)
        self.handle.write(message + '\n')

    def close(self):
        self.handle.close()


def stage_training_data(images, root, category='platform_category'):
    target = Path(root) / category / 'train' / 'good'
    target.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(images):
        destination = target / f'{index:06d}{source.suffix.lower()}'
        try:
            os.symlink(source.resolve(), destination)
        except OSError:
            shutil.copyfile(source, destination)
    return Path(root), category


def learning_rate(step, total, base, final, warmup):
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return final + .5 * (base-final) * (1 + math.cos(math.pi * min(1., progress)))


def gpu_memory_mb():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=used_memory', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True, timeout=5)
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        return max(values, default=0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def run(args):
    input_root, output_root = Path(args.input_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    state = StateLog(output_root / 'state.txt')
    started = time.perf_counter()
    try:
        data_root = input_root / 'original' / 'input_data'
        param_path = data_root / 'param.json'
        if not param_path.is_file():
            param_path = input_root / 'param.json'
        params = load_json(param_path) if param_path.is_file() else {}
        image_root = data_root / 'images' / 'train'
        images = discover_images(image_root)
        batch = bounded_number(params, ('batch_size', 'batchSize', '批大小'), 4, int, 1, 64)
        workers = bounded_number(params, ('num_workers', 'numWorkers', '线程数'), 4, int, 0, 32)
        raw_epochs = get_parameter(params, ('epochs', 'epoch', '迭代轮次'))
        explicit_iters = get_parameter(params, ('total_iters', 'totalIters', 'iterations', '迭代次数'))
        epochs = bounded_number(params, ('epochs', 'epoch', '迭代轮次'),
                                1 if explicit_iters is not None and raw_epochs is None else 100,
                                int, 1, 10000)
        total_iters = (bounded_number(params, ('total_iters', 'totalIters', 'iterations', '迭代次数'),
                                      10000, int, 1, 1000000) if explicit_iters is not None
                       else epochs * max(1, math.ceil(len(images) / batch)))
        warmup = min(100, max(1, total_iters // 10))
        mining_warmup = min(1000, total_iters)
        crop = bounded_number(params, ('crop_size', 'cropSize', 'resolution', '分辨率'), 560, int, 224, 1120)
        if crop % 14:
            raise ValueError('crop_size/resolution must be divisible by 14')
        lr = bounded_number(params, ('lr', 'learningRate', '学习率'), .002, float, 1e-7, 1.)
        final_lr = bounded_number(params, ('final_lr', 'finalLr'), lr/10, float, 0., lr)
        augment = parse_bool(get_parameter(params, ('train_augment', 'trainAugment', '数据增强'), True), 'train_augment')
        log_every = max(1, min(100, total_iters // 100 or 1))
        runtime = Path(args.runtime_dir)
        if runtime.exists():
            shutil.rmtree(runtime)
        staged, category = stage_training_data(images, runtime / 'dataset')
        work = runtime / 'train_output'
        command = [sys.executable, '-u', str(Path(__file__).with_name('dinomaly_omniad_uni.py')),
                   '--mode', 'train', '--data_path', str(staged), '--categories', category,
                   '--output_dir', str(work), '--preprocess', 'letterbox', '--image_size', str(crop),
                   '--crop_size', str(crop), '--eval_mask_size', str(crop), '--batch_size', str(batch),
                   '--num_workers', str(workers), '--total_iters', str(total_iters), '--lr', str(lr),
                   '--final_lr', str(final_lr), '--warmup_iters', str(warmup),
                   '--hm_warmup_iters', str(mining_warmup), '--log_every', str(log_every), '--eval_every', '0',
                   '--device', args.device, '--train_augment' if augment else '--no-train_augment']
        state.write(f'Epoch (train) [0/{epochs}] Iter [0/{total_iters}] lr {lr:.8g} '
                    f'eta unknown time 0.000 memory 0 loss 0.000000')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        assert process.stdout is not None
        last = time.perf_counter()
        for line in process.stdout:
            line = line.rstrip()
            print(line, flush=True)
            match = ITERATION.search(line)
            if match:
                step, total, loss = int(match.group(1)), int(match.group(2)), float(match.group(3))
                now = time.perf_counter()
                elapsed = now - started
                eta = elapsed / max(1, step) * (total-step)
                memory = gpu_memory_mb()
                epoch = min(epochs, math.ceil(step / max(1, total/epochs)))
                state.write(f'Epoch (train) [{epoch}/{epochs}] Iter [{step}/{total}] '
                            f'lr {learning_rate(step, total, lr, final_lr, warmup):.8g} '
                            f'eta {eta:.1f} time {now-last:.3f} memory {memory} loss {loss:.6f}')
                last = now
        code = process.wait()
        if code:
            raise RuntimeError(f'training process exited with code {code}')
        checkpoint = work / 'omniad_dinomaly_uni.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError('training finished without a checkpoint')
        temporary = output_root / 'output.bin.tmp'
        shutil.copyfile(checkpoint, temporary)
        os.replace(temporary, output_root / 'output.bin')
        state.write(f'Epoch (train) [{epochs}/{epochs}] Iter [{total_iters}/{total_iters}] '
                    f'lr {final_lr:.8g} eta 0 time {time.perf_counter()-started:.3f} '
                    f'memory 0 loss 0 finish')
    except Exception as exc:
        state.write(f'Epoch (train) [0/0] Iter [0/0] lr 0 eta 0 time '
                    f'{time.perf_counter()-started:.3f} memory 0 loss nan error {type(exc).__name__}: {exc}')
        raise
    finally:
        state.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='/input')
    parser.add_argument('--output_dir', default='/output')
    parser.add_argument('--runtime_dir', default='/tmp/omniad_platform_train')
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

"""Prepare and validate a real platform training/inference release test."""

import argparse
import json
import math
from pathlib import Path
import shutil


SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(dataset, work):
    if not dataset:
        raise ValueError('--dataset must point to the product category directory')
    source = Path(dataset).resolve()
    train = sorted(p for p in (source / 'train/good').rglob('*')
                   if p.is_file() and p.suffix.lower() in SUFFIXES)
    test = sorted(p for p in (source / 'test').rglob('*')
                  if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not train or not test:
        raise ValueError('Both train/good and test images are required')
    for name in ('train-input', 'train-output', 'infer-input', 'infer-output'):
        (work / name).mkdir(parents=True, exist_ok=False)
    train_root = work / 'train-input/original/input_data'
    for group, destination in ((train, train_root / 'images/train'),
                               (test, work / 'infer-input/imgs')):
        destination.mkdir(parents=True)
        for index, path in enumerate(group):
            shutil.copyfile(path, destination / f'{index:05d}{path.suffix.lower()}')
    write_json(train_root / 'param.json', dict(epochs=100, batch_size=4,
               num_workers=4, resolution=512, train_augment=True))
    write_json(work / 'infer-input/param.json', dict(algorithmType=103, algorithmSubType=0,
               modelType=1, modelPath='/model/output.bin', imagePath='/imgs', platType=2,
               cnnParam=dict(MinScore=.1, pixelThreshold=.12, minArea=4)))
    write_json(work / 'expected.json', dict(train_images=len(train), test_images=len(test),
               iterations=100 * math.ceil(len(train) / 4),
               source_images=[str(p.relative_to(source)) for p in test]))
    print(f'Prepared {len(train)} normal training images and {len(test)} test images', flush=True)


def check_training(work):
    expected = read_json(work / 'expected.json')
    output = work / 'train-output'
    assert 'omniad bootstrap exit code=0' in (output / 'bootstrap.log').read_text(encoding='utf-8')
    lines = (output / 'state.txt').read_text(encoding='utf-8').splitlines()
    assert lines[-1] == 'finish omniad training'
    assert not any('finish' in line for line in lines[:-1])
    total = expected['iterations']
    assert any(f'Iter: {total}/{total} ' in line for line in lines)
    config = read_json(output / 'training_config.json')
    assert config['requested_resolution'] == 512 and config['effective_resolution'] == 518
    assert config['train_images'] == expected['train_images']
    assert config['total_iters'] == total
    assert (output / 'output.bin').stat().st_size > 0
    model = work / 'infer-input/model/output.bin'
    model.parent.mkdir()
    shutil.copyfile(output / 'output.bin', model)
    print(f'Training passed: {total} iterations, resolution 512 -> 518, output.bin exported', flush=True)


def check_inference(work):
    import numpy as np
    from PIL import Image

    expected = read_json(work / 'expected.json')
    assert 'omniad bootstrap exit code=0' in (work / 'infer-output/bootstrap.log').read_text(encoding='utf-8')
    assert 'omniad shell start mode=infer' in (work / 'infer-output/entrypoint.log').read_text(encoding='utf-8')
    pred_root = work / 'infer-input/pred'
    predictions = read_json(pred_root / 'pred.json')
    images = sorted((work / 'infer-input/imgs').iterdir())
    assert len(images) == len(predictions) == expected['test_images']
    assert set(predictions) == {f'test/{image.name}' for image in images}
    for path in images:
        record = predictions[f'test/{path.name}']
        assert record['anomaly_map'] == f'pred_maps/test/{path.stem}.npy'
        score = record['anomaly_score']
        assert math.isfinite(score) and 0 <= score <= 1
        anomaly = np.load(pred_root / record['anomaly_map'], allow_pickle=False)
        with Image.open(path) as image:
            assert anomaly.shape == (image.height, image.width)
        assert anomaly.dtype == np.float32 and np.isfinite(anomaly).all()
        assert anomaly.min() >= 0 and anomaly.max() <= 1
        objects = read_json(work / f'infer-output/{path.stem}.json')
        assert isinstance(objects, list)
        for obj in objects:
            assert {'category_Name', 'score', 'id', 'segmentation', 'type'} <= obj.keys()
            assert obj['type'] == 'polygon'
    lines = (work / 'infer-output/reasoning.log').read_text(encoding='utf-8').splitlines()
    assert lines[0] == 'reasoning start' and lines[-1] == 'reasoning close success'
    assert lines.count('reasoning start') == 1 and not any('reasoning error,' in line for line in lines)
    assert sum(line.startswith('reasoning imageName=') for line in lines) == len(images)
    write_json(work / 'verification.json', dict(status='passed', train_images=expected['train_images'],
               training_iterations=expected['iterations'], test_images=len(images),
               requested_resolution=512, effective_resolution=518,
               note='Protocol and execution checks only; not an accuracy evaluation.'))
    print(f'Inference passed: {len(images)} original-size float32 maps and visualization JSON files', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check-training', 'check-inference'))
    parser.add_argument('--work', required=True)
    parser.add_argument('--dataset')
    args = parser.parse_args()
    work = Path(args.work).resolve()
    if args.action == 'prepare':
        prepare(args.dataset, work)
    elif args.action == 'check-training':
        check_training(work)
    else:
        check_inference(work)


if __name__ == '__main__':
    main()

# AI training platform Docker package

This image supports both platform jobs. `train` consumes normal images and emits
`/output/output.bin`; `infer` loads that file and emits visualization JSON, metric
maps and the mandatory reasoning log. Dataset images are never copied into the
image. The public official DINOv2 register ViT-B/14 backbone is downloaded during
the Docker build, so training and inference do not require outbound network access.

## Build and smoke checks

Build on an x86-64 Linux machine with Docker 28.1 (or a compatible recent Docker):

```bash
docker build --pull -t omniad_dinomaly:1.0.0 .
docker image inspect omniad_dinomaly:1.0.0
```

The base image is PyTorch 2.3.1 with CUDA 11.8 and cuDNN 8 on Ubuntu. The target
RTX 4090 driver 570 is backward compatible with this CUDA runtime. The trained
state dictionary is compatible with the project's PyTorch 2.4.1 development runs.
The platform must use the NVIDIA runtime for GPU jobs.

Before upload, exercise the exact mounts. Training input must contain:

```text
train-input/original/input_data/
  param.json
  images/train/
    000.png
    ...
```

```bash
docker run --rm --runtime=nvidia --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -v /dev/shm:/dev/shm \
  -v "$PWD/train-input:/input" \
  -v "$PWD/train-output:/output" \
  omniad_dinomaly:1.0.0 train

test -s train-output/output.bin
grep finish train-output/state.txt
```

Inference input example:

```text
infer-input/
  param.json
  models/output.bin
  imgs/000.png
  pred/gt.json              # supplied by the platform; never overwritten
```

```bash
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -v /tmp/:/tmp/ \
  -v "$PWD/infer-input:/input" \
  -v "$PWD/infer-output:/output" \
  omniad_dinomaly:1.0.0 infer

test -s infer-input/pred/pred.json
test -s infer-input/pred/pred_maps/test/000.npy
grep "reasoning close success" infer-output/reasoning.log
```

`platform_examples/` contains parameter templates. `imagePath` may name one image
or a directory below `/input`. Flat platform output requires unique file names and
stems. The adapter rejects paths escaping `/input`, encrypted models, duplicate
names, unsupported algorithm subtypes, invalid values and GPU requests when CUDA
is unavailable. Errors are written in the required log format before a nonzero exit.

Metric anomaly maps are float32, original image size, finite and clipped to [0,1].
They are not normalized from each test image. `MinScore`, `pixelThreshold` and
`minArea` affect only `/output/<stem>.json`; they do not alter metric maps or image
scores. `/input/pred/gt.json` is left untouched. Image-level score is the mean of
the highest 1% model-map pixels. Each image's end-to-end and SDK times are logged.

## Platform command templates

Use the platform's own variable syntax exactly as configured by its administrator.
Do not put literal host paths or GPU IDs into the image.

Training command template:

```bash
docker run --runtime=nvidia --rm --name ${containerName} --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=${gpuNumber} \
  -v /dev/shm:/dev/shm \
  -v ${inputDir}:/input \
  -v ${outputDir}:/output \
  ${imageVersion} train
```

Inference command template:

```bash
docker run --runtime=nvidia --rm --name ${taskName} \
  -e NVIDIA_VISIBLE_DEVICES=${visibleDevice} \
  -v /tmp/:/tmp/ \
  -v ${inputPath}:/input \
  -v ${outputPath}:/output \
  ${tagName} infer
```

Suggested management values: third-party image, algorithm subtype `103` / unsupervised
segmentation, GPU supported, custom parameters supported, non-resident image. Use
the exact image name/version required by the account's naming convention.

## Upload archive

The upload UI requests a ZIP file with a 5 GB maximum. Save the Docker image, ZIP
the resulting tar, and verify it can be loaded before uploading:

```bash
docker save -o omniad_dinomaly_1.0.0.tar omniad_dinomaly:1.0.0
zip -1 omniad_dinomaly_1.0.0.zip omniad_dinomaly_1.0.0.tar
ls -lh omniad_dinomaly_1.0.0.zip

unzip -t omniad_dinomaly_1.0.0.zip
unzip -p omniad_dinomaly_1.0.0.zip omniad_dinomaly_1.0.0.tar | docker load
```

Do not include the dataset, development predictions, diagnostics, or trained local
checkpoints in the build context. `.dockerignore` excludes those paths and test
files. If the ZIP exceeds 5 GB, stop rather than splitting it: inspect Docker layers
and use the platform's documented base-image mechanism if available.

## Training semantics and limits

Only `/input/original/input_data/images/train` is used for optimization. Files are
staged under `/tmp` into the repository's normal-only layout. With `epochs`, total
optimizer iterations are `epochs * ceil(normal_images / batch_size)`; an explicit
`total_iters` overrides this. Resolution must be divisible by the DINOv2 patch size
14. The log always includes `Epoch (train)`, `Iter`, `lr`, `eta`, `time`, `memory`,
`loss`, and the terminal `finish` marker. Model export is atomic and fixed to
`/output/output.bin`.

This platform package is a clean single-model Dinomaly training/inference path.
The development-only routed ensembles, category-specific memory banks, full-coverage
multi-view experiments and label-guided result selection are intentionally absent:
they are not self-contained in `output.bin`, and multi-view retrieval threatens the
100 ms inference limit. Consequently, the Docker package is protocol-complete but
does not automatically reproduce the best routed development score. Measure RTX4090
latency on the actual platform; the competition gives zero inference-time points
above 100 ms and requires peak VRAM below 24 GB.

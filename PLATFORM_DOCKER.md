# AI training platform Docker package

The image contains one Dinomaly model path for both platform training and inference.
Training reads normal images from `/input/original/input_data/images/train` and writes
`/output/output.bin`. Inference reads `/input/param.json`, writes one visualization
JSON per image under `/output`, writes metric files below `/input/pred`, and records
`/output/reasoning.log`.

The image is based on PyTorch 2.3.1, CUDA 11.8 and cuDNN 8. The public DINOv2
register ViT-B/14 backbone is downloaded during `docker build`, so platform jobs do
not require outbound network access.

## Build

Build and save the image on x86-64 Linux:

```bash
docker build --pull -t omniad_dinomaly:1.0.0 .
docker image inspect omniad_dinomaly:1.0.0
docker save -o omniad_dinomaly_1.0.0.tar omniad_dinomaly:1.0.0
zip -1 omniad_dinomaly_1.0.0.zip omniad_dinomaly_1.0.0.tar
unzip -t omniad_dinomaly_1.0.0.zip
```

The platform upload limit is 5 GB. Check the final ZIP before upload. Dataset files,
development predictions, diagnostics and local checkpoints are excluded by
`.dockerignore`.

## Training protocol

Expected input:

```text
/input/original/input_data/
  param.json
  images/train/
    000.png
    ...
```

Supported training parameters are shown in `platform_examples/train_param.json`.
Only OK/normal images in `images/train` are used. The fixed outputs are:

```text
/output/output.bin
/output/state.txt
```

`state.txt` is UTF-8. A progress line follows the platform parser format exactly:

```text
Epoch(train) [14][7/7] Iter: 98/140 lr:0.000094  eta:0:00:08  time:0.131060  memory:1975  loss:24.520100
```

The word `finish` is written only after `output.bin` has been exported successfully.

Use this platform command template:

```bash
docker run \
  --cap-add=ALL \
  --gpus '"device=${gpuNumber}"' \
  --name=${containerName} \
  --security-opt seccomp=unconfined \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /dev/shm:/dev/shm \
  -v ${inputDir}:/input \
  -v ${outputDir}:/output \
  --entrypoint /bin/bash \
  --rm ${imageVersion} \
  -c "cd /opt/omniad && sh train.sh /input/ /output/"
```

## Inference protocol

Expected input:

```text
/input/
  param.json
  model/output.bin
  imgs/000.png
  pred/gt.json
```

The parameter schema is shown in `platform_examples/infer_param.json`. Required
values are `algorithmType=103`, `algorithmSubType=0`, `modelType`, `modelPath`,
`imagePath`, and `platType` (`1` for CPU or `2` for GPU). Paths such as
`/model/output.bin` and `/imgs/` are resolved inside the `/input` mount. Custom
parameters including `MinScore`, `pixelThreshold`, and `minArea` are read directly
from `cnnParam`; `cnnParam.extendParamMap` is also accepted for compatibility.

For every input image, the adapter writes `/output/<image-stem>.json`. A normal
result is `[]`. Each detected polygon uses this schema:

```json
[
  {
    "category_Name": "1",
    "score": 0.9691603,
    "id": 1,
    "segmentation": [[1019, 247, 1020, 248, 1020, 250]],
    "type": "polygon"
  }
]
```

Metric outputs are written to the mandatory locations:

```text
/input/pred/pred.json
/input/pred/gt.json
/input/pred/pred_maps/test/000.npy
```

The `pred.json` key is `test/000.png`; `anomaly_map` is
`pred_maps/test/000.npy`. Maps are float32 at original image size, finite, and
clipped to `[0, 1]`. Existing `gt.json` is never overwritten.

`/output/reasoning.log` contains the required records without spaces around `=`:

```text
reasoning start
reasoning imageName=000.png,sequence=1,algRunTime=153.000000,sdkRunTime=140.000000
reasoning close success
```

Failures are recorded as:

```text
reasoning error, code=0x80100000, message=ValueError: error details
```

Use this platform command template:

```bash
docker run --runtime=nvidia --cap-add=ALL \
  --env NVIDIA_VISIBLE_DEVICES=${visibleDevice} \
  --name=${taskName} \
  -v /tmp/:/tmp/ \
  -v ${inputPath}:/input \
  -v ${outputPath}:/output \
  --rm ${tagName} /bin/bash \
  -c 'cd /opt/omniad && sh start.sh /input/ /output/'
```

Suggested management settings: third-party image, algorithm type `103` (unsupervised
segmentation), GPU supported, custom parameters supported, and non-resident image.
The image intentionally has no `ENTRYPOINT`, because the platform appends
`/bin/bash -c ...` to the image command.

## Local smoke checks

Run training with the same entry command used by the platform:

```bash
docker run --rm --gpus '"device=0"' --security-opt seccomp=unconfined \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /dev/shm:/dev/shm \
  -v "$PWD/train-input:/input" -v "$PWD/train-output:/output" \
  --entrypoint /bin/bash omniad_dinomaly:1.0.0 \
  -c 'cd /opt/omniad && sh train.sh /input/ /output/'
test -s train-output/output.bin
tail -n 1 train-output/state.txt | grep '^finish'
```

Run inference:

```bash
docker run --rm --runtime=nvidia --cap-add=ALL \
  -e NVIDIA_VISIBLE_DEVICES=0 -v /tmp/:/tmp/ \
  -v "$PWD/infer-input:/input" -v "$PWD/infer-output:/output" \
  omniad_dinomaly:1.0.0 /bin/bash \
  -c 'cd /opt/omniad && sh start.sh /input/ /output/'
test -s infer-input/pred/pred.json
test -s infer-input/pred/pred_maps/test/000.npy
tail -n 1 infer-output/reasoning.log | grep 'reasoning close success'
```

The package adapter is protocol-complete, but the image size, RTX 4090 runtime,
peak VRAM, and per-image latency must still be verified by building and running the
image on a Linux NVIDIA host. Encrypted model files are not supported; leave the
optional `modelPassword` field empty.

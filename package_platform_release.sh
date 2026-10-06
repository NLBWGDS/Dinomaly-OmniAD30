#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
repo="$PWD"
dataset="${1:?Pass the product directory containing train/good and test}"
release="${2:?Pass a new, non-existing release directory}"
image="omniad_dinomaly:1.0.2"
platform_image="${PLATFORM_IMAGE:-zhejiang_ai_competition:hikrobot_comp_40_v2}"
gpu="${RELEASE_GPU:-0}"
mkdir "$release"
release="$(cd "$release" && pwd)"
trap 'result=$?; printf "%s\n" "$result" > "$release/exit_code.txt"' EXIT

python3 verify_platform_release.py prepare --dataset "$dataset" --work "$release"
docker build -f Dockerfile.platform-update -t "$image" . 2>&1 | tee "$release/build.log"
docker tag "$image" "$platform_image"
docker image inspect "$image" > "$release/image_inspect.json"

# Exercise the PDF's default training command, real GPU, and offline backbone.
docker run --rm --network none --gpus "device=$gpu" \
  --name "omniad-release-train-$$" --security-opt seccomp=unconfined \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v /dev/shm:/dev/shm \
  -v "$release/train-input:/input" -v "$release/train-output:/output" \
  --entrypoint /bin/bash "$image" \
  -c 'cd ./root && python train.py' 2>&1 | tee "$release/train_container.log"
python3 verify_platform_release.py check-training --work "$release"

docker run --rm --network none --runtime=nvidia \
  --env "NVIDIA_VISIBLE_DEVICES=$gpu" --name "omniad-release-infer-$$" \
  -v /tmp/:/tmp/ -v "$release/infer-input:/input" -v "$release/infer-output:/output" \
  "$image" /bin/bash -c 'cd /opt/omniad && sh start.sh /input/ /output/' \
  2>&1 | tee "$release/infer_container.log"
docker run --rm --network none -v "$release:/release" \
  -v "$repo/verify_platform_release.py:/verify_release.py:ro" "$image" \
  python /verify_release.py check-inference --work /release

docker save -o "$release/omniad_dinomaly_1.0.2.tar" "$image" "$platform_image"
zip -1 -j "$release/omniad_dinomaly_1.0.2.zip" "$release/omniad_dinomaly_1.0.2.tar"
unzip -t "$release/omniad_dinomaly_1.0.2.zip" | tee "$release/zip_test.log"
cd "$release"
sha256sum omniad_dinomaly_1.0.2.zip | tee SHA256SUMS
bytes="$(stat -c %s omniad_dinomaly_1.0.2.zip)"
printf 'Upload image tag: %s\nZIP bytes: %s\n' "$platform_image" "$bytes" | tee upload_info.txt
if [ "$bytes" -ge 5000000000 ]; then
  echo 'ZIP exceeds the conservative 5 GB upload limit' >&2
  exit 1
fi
echo "Release ready: $release/omniad_dinomaly_1.0.2.zip"

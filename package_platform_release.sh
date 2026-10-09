#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
repo="$PWD"
dataset="${1:?Pass the product directory containing train/good and test}"
release="${2:?Pass a new, non-existing release directory}"
image="omniad_dinomaly:1.0.3"
platform_image="${PLATFORM_IMAGE:-zhejiang_ai_competition:hikrobot_comp_40_v3}"
gpu="${RELEASE_GPU:-0}"
mkdir "$release"
release="$(cd "$release" && pwd)"
trap 'result=$?; printf "%s\n" "$result" > "$release/exit_code.txt"' EXIT

python3 verify_platform_release.py prepare --dataset "$dataset" --work "$release"
docker build -f Dockerfile.platform-update -t "$image" . 2>&1 | tee "$release/build.log"
docker tag "$image" "$platform_image"
docker image inspect "$image" > "$release/image_inspect.json"
python3 - "$release/image_inspect.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as handle:
    image = json.load(handle)[0]
assert (image['Os'], image['Architecture']) == ('linux', 'amd64')
assert not image['Config'].get('Entrypoint'), 'Platform appends its own shell command'
assert image['Config']['WorkingDir'] == '/opt/omniad'
PY

# Prove that failures at each supported entry point produce persistent logs.
docker run --rm --network none -v "$release:/release" "$image" \
  python /opt/omniad/verify_platform_startup.py --output /release/startup-checks \
  2>&1 | tee "$release/startup_container.log"

# Exercise the PDF's default training command, real GPU, and offline backbone.
docker run --rm --network none --cap-add=ALL --gpus "device=$gpu" \
  --name "omniad-release-train-$$" --security-opt seccomp=unconfined \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v /dev/shm:/dev/shm \
  -v "$release/train-input:/input" -v "$release/train-output:/output" \
  --entrypoint /bin/bash "$image" \
  -c 'cd ./root && python train.py' 2>&1 | tee "$release/train_container.log"
python3 verify_platform_release.py check-training --work "$release" \
  2>&1 | tee "$release/train_verification.log"

docker run --rm --network none --runtime=nvidia --cap-add=ALL \
  --env "NVIDIA_VISIBLE_DEVICES=$gpu" --name "omniad-release-infer-$$" \
  -v /tmp/:/tmp/ -v "$release/infer-input:/input" -v "$release/infer-output:/output" \
  "$image" /bin/bash -c 'cd /home/apps/ats-reasoning-tool/ && sh start.sh /input/ /output/' \
  2>&1 | tee "$release/infer_container.log"
docker run --rm --network none -v "$release:/release" \
  -v "$repo/verify_platform_release.py:/verify_release.py:ro" "$image" \
  python /verify_release.py check-inference --work /release \
  2>&1 | tee "$release/infer_verification.log"

docker save -o "$release/omniad_dinomaly_1.0.3.tar" "$platform_image"
python3 - "$release/omniad_dinomaly_1.0.3.tar" "$platform_image" <<'PY'
import json
import sys
import tarfile

archive, expected_tag = sys.argv[1:]
with tarfile.open(archive, 'r') as saved:
    manifest = json.load(saved.extractfile('manifest.json'))
tags = [tag for item in manifest for tag in (item.get('RepoTags') or [])]
if tags != [expected_tag]:
    raise SystemExit(f'Expected one platform tag {expected_tag!r}, found {tags!r}')
print(f'Docker archive tag verified: {expected_tag}', flush=True)
PY
before_load="$(docker image inspect --format '{{.Id}}' "$platform_image")"
docker load -i "$release/omniad_dinomaly_1.0.3.tar" 2>&1 | tee "$release/load_test.log"
test "$(docker image inspect --format '{{.Id}}' "$platform_image")" = "$before_load"
zip -1 -j "$release/omniad_dinomaly_1.0.3.zip" "$release/omniad_dinomaly_1.0.3.tar"
unzip -t "$release/omniad_dinomaly_1.0.3.zip" | tee "$release/zip_test.log"
cd "$release"
sha256sum omniad_dinomaly_1.0.3.zip | tee SHA256SUMS
bytes="$(stat -c %s omniad_dinomaly_1.0.3.zip)"
printf 'Upload image tag: %s\nZIP bytes: %s\n' "$platform_image" "$bytes" | tee upload_info.txt
if [ "$bytes" -ge 5000000000 ]; then
  echo 'ZIP exceeds the conservative 5 GB upload limit' >&2
  exit 1
fi
echo "Release ready: $release/omniad_dinomaly_1.0.3.zip"

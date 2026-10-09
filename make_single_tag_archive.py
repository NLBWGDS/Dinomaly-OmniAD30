"""Rewrite a Docker save ZIP so it contains exactly one repository tag."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile


def rewrite_json(name, data, expected_tag):
    value = json.loads(data)
    if name == 'manifest.json':
        if len(value) != 1:
            raise ValueError(f'Expected one image manifest, found {len(value)}')
        value[0]['RepoTags'] = [expected_tag]
    elif name == 'index.json':
        manifests = value.get('manifests') or []
        if not manifests:
            raise ValueError('index.json contains no manifests')
        selected = None
        expected_name = f'docker.io/library/{expected_tag}'
        for manifest in manifests:
            annotations = manifest.get('annotations') or {}
            if annotations.get('io.containerd.image.name') == expected_name:
                selected = manifest
                break
        if selected is None:
            selected = dict(manifests[0])
            annotations = dict(selected.get('annotations') or {})
            annotations['io.containerd.image.name'] = expected_name
            annotations['org.opencontainers.image.ref.name'] = expected_tag.rsplit(':', 1)[-1]
            selected['annotations'] = annotations
        value['manifests'] = [selected]
    return json.dumps(value, separators=(',', ':')).encode('utf-8')


def rewrite_tar(source, destination, expected_tag):
    seen = set()
    with tarfile.open(fileobj=source, mode='r|') as reader:
        with tarfile.open(fileobj=destination, mode='w|') as writer:
            for member in reader:
                stream = reader.extractfile(member) if member.isfile() else None
                if member.name in {'manifest.json', 'index.json'}:
                    data = rewrite_json(member.name, stream.read(), expected_tag)
                    member.size = len(data)
                    stream = io.BytesIO(data)
                    seen.add(member.name)
                writer.addfile(member, stream)
    missing = {'manifest.json', 'index.json'} - seen
    if missing:
        raise ValueError(f'Docker archive is missing {sorted(missing)}')


def inspect_tar(path, expected_tag):
    with tarfile.open(path, 'r') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        index = json.load(archive.extractfile('index.json'))
    tags = [tag for item in manifest for tag in (item.get('RepoTags') or [])]
    names = [
        (item.get('annotations') or {}).get('io.containerd.image.name')
        for item in index.get('manifests') or []
    ]
    expected_name = f'docker.io/library/{expected_tag}'
    if tags != [expected_tag] or names != [expected_name]:
        raise ValueError(f'Unexpected tags after rewrite: RepoTags={tags}, index names={names}')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_zip', type=Path)
    parser.add_argument('output_zip', type=Path)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()

    source_zip = args.source_zip.resolve()
    output_zip = args.output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_tar = output_zip.with_suffix('.tar')
    if output_zip.exists() or output_tar.exists():
        raise FileExistsError('Output ZIP or intermediate TAR already exists')

    with zipfile.ZipFile(source_zip) as package:
        package.testzip()
        tar_members = [item for item in package.infolist() if item.filename.endswith('.tar')]
        if len(tar_members) != 1:
            raise ValueError(f'Expected one TAR in ZIP, found {len(tar_members)}')
        with package.open(tar_members[0]) as source, output_tar.open('wb') as destination:
            rewrite_tar(source, destination, args.tag)

    inspect_tar(output_tar, args.tag)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as package:
        package.write(output_tar, output_tar.name)
    with zipfile.ZipFile(output_zip) as package:
        if package.testzip() is not None:
            raise ValueError('Output ZIP CRC test failed')
    output_tar.unlink()
    print(json.dumps({
        'output': str(output_zip),
        'bytes': output_zip.stat().st_size,
        'sha256': sha256(output_zip),
        'tag': args.tag,
    }, indent=2))


if __name__ == '__main__':
    main()

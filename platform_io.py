"""Shared, dependency-light helpers for the competition platform adapters."""

import json
import os
from pathlib import Path


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def load_json(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def parameter_views(document):
    """Return dictionaries in precedence order, including extendParamMap."""
    views = []
    cnn = document.get('cnnParam')
    if isinstance(cnn, dict):
        extend = cnn.get('extendParamMap')
        if isinstance(extend, dict):
            views.append(extend)
        views.append(cnn)
    extend = document.get('extendParamMap')
    if isinstance(extend, dict):
        views.append(extend)
    views.append(document)
    return views


def get_parameter(document, aliases, default=None):
    wanted = {str(alias).casefold() for alias in aliases}
    for view in parameter_views(document):
        folded = {str(key).casefold(): value for key, value in view.items()}
        for alias in wanted:
            if alias in folded and folded[alias] not in (None, ''):
                return folded[alias]
    return default


def parse_bool(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {'1', 'true', 'yes', 'on', '是', '开启'}:
        return True
    if text in {'0', 'false', 'no', 'off', '否', '关闭'}:
        return False
    raise ValueError(f'{name} must be boolean')


def bounded_number(document, aliases, default, cast, minimum, maximum):
    raw = get_parameter(document, aliases, default)
    try:
        value = cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{aliases[0]} must be a number') from exc
    if not minimum <= value <= maximum:
        raise ValueError(f'{aliases[0]} must be in [{minimum}, {maximum}]')
    return value


def contained_path(raw, root, name, must_exist=True):
    root = Path(root).resolve()
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f'{name} must stay under {root}')
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f'{name} does not exist: {candidate}')
    return candidate


def platform_input_path(raw, root, name, must_exist=True):
    """Resolve platform virtual /model and /imgs paths below the /input mount."""
    root = Path(root).resolve()
    direct = Path(str(raw))
    if direct.is_absolute():
        resolved = direct.resolve()
        if resolved == root or root in resolved.parents:
            return contained_path(resolved, root, name, must_exist)
    portable = str(raw).replace('\\', '/')
    if portable == '/input':
        portable = ''
    elif portable.startswith('/input/'):
        portable = portable[len('/input/'):]
    else:
        portable = portable.lstrip('/')
    return contained_path(portable, root, name, must_exist)


def discover_images(path):
    path = Path(path)
    if path.is_file():
        images = [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    else:
        images = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f'No supported images found under {path}')
    names = [image.name for image in images]
    stems = [image.stem for image in images]
    if len(names) != len(set(names)) or len(stems) != len(set(stems)):
        raise ValueError('Platform output is flat; image names and stems must be unique')
    return images


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

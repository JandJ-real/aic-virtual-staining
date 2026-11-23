"""Shared CPU-only version, image/metric and NumPy/Torch bridge preflight."""
from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re

from common import ROOT, sha256

RANGES = {
    'python': ((3, 9, 0), (4, 0, 0)),
    'torch': ((2, 4, 0), (2, 6, 0)),
    'numpy': ((1, 20, 3), (3, 0, 0)),
    'Pillow': ((8, 4, 0), (12, 0, 0)),
    'scikit-image': ((0, 18, 3), (0, 25, 0)),
    'scipy': ((1, 0, 1), (2, 0, 0)),
}


def runtime_versions():
    result = {'python': platform.python_version()}
    for name in RANGES:
        if name == 'python':
            continue
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError('Missing dependency '+name+'; do not auto-upgrade the CUDA environment') from error
    return result


def validate_versions(versions):
    problems = []
    for name, (lower, upper) in RANGES.items():
        match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', versions.get(name, ''))
        parsed = tuple(int(part or 0) for part in match.groups()) if match else None
        if parsed is None or not lower <= parsed < upper:
            problems.append('%s=%s (requires >=%s, <%s)' %
                (name, versions.get(name, 'missing'), '.'.join(map(str, lower)), '.'.join(map(str, upper))))
    if problems:
        raise RuntimeError('Unsupported runtime: '+'; '.join(problems))


def check_metric_contract():
    import numpy as np
    from metric_cases import cases
    from metrics import image_metrics, png_pixels
    path = ROOT/'metric_goldens.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    fixtures = cases()
    if set(frozen['cases']) != {name for name, _, _ in fixtures}:
        raise RuntimeError('Metric fixture identity mismatch')
    maximum = 0.0
    for name, target, prediction in fixtures:
        actual = image_metrics(target, prediction)
        for field, wanted in frozen['cases'][name].items():
            value = actual[field]
            if wanted == 'inf':
                if value != float('inf'):
                    raise RuntimeError('Perfect-image PSNR contract mismatch')
                continue
            delta = abs(value-wanted)
            if not math.isfinite(value) or delta > frozen['absolute_tolerance']:
                raise RuntimeError('Metric contract mismatch: %s/%s delta=%s' % (name, field, delta))
            maximum = max(maximum, delta)
    encoded, decoded = png_pixels(np.full((256, 256), .5, dtype=np.float32))
    if (not encoded.startswith(b'\x89PNG\r\n\x1a\n') or decoded.dtype != np.uint8
            or decoded.shape != (256, 256, 3) or not np.all(decoded == 128)):
        raise RuntimeError('Fixed uint8 RGB PNG roundtrip mismatch')
    return dict(cases=len(fixtures), max_absolute_delta=maximum, png_roundtrip=True,
                reference='frozen original0.24 RGB API synthetic samples',
                baseline_sha256=sha256(path), official_images_used=False)


def check_runtime():
    versions = runtime_versions()
    validate_versions(versions)
    import numpy as np
    import torch
    # CPU only: catch NumPy ABI/bridge problems without initializing CUDA.
    original = np.array([0, 1, 2], dtype=np.float32)
    tensor = torch.from_numpy(original)
    if not np.array_equal((tensor+1).numpy(), np.array([1, 2, 3], dtype=np.float32)):
        raise RuntimeError('NumPy/Torch CPU bridge mismatch')
    return dict(versions=versions, metric=check_metric_contract(), numpy_torch_bridge=True,
                cuda_checked=False)

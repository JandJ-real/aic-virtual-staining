"""Small, torch-free contract shared by login and compute entrypoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORGANS = ('colon', 'liver', 'stomach')
MARKERS = ('CD68', 'CD45RO')
COUNTS = dict(zip(ORGANS, (1, 2, 5)))
WEIGHTS = dict(zip(ORGANS, (0.1, 0.2, 0.7)))
DATA_SHA = 'd7e66cc07facc289227a214897e93a6908242b2bbdb017a097b15156bb772e4d'
WEIGHT_SHA = '4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd'
REAL_WEIGHT_SHA = '4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387'
PUBLIC_WEIGHTS = {
    'gaussian': dict(file='pretrained/gaussian_color_denoising_blind.pth', sha256=WEIGHT_SHA),
    'real': dict(file='pretrained/real_denoising.pth', sha256=REAL_WEIGHT_SHA),
}
INITIALIZATIONS = {'CD68': 'gaussian', 'CD45RO': 'gaussian'}


def initialization(marker):
    if marker not in MARKERS:
        raise ValueError('Unsupported experiment marker')
    name = INITIALIZATIONS[marker]
    return dict(PUBLIC_WEIGHTS[name], name=name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'inf' if value > 0 else '-inf' if value < 0 else 'nan'
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def write_text_lf(path, text, encoding='utf-8'):
    with Path(path).open('w', encoding=encoding, newline='\n') as stream:
        stream.write(text)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    write_text_lf(temp, json.dumps(safe_json(value), ensure_ascii=False,
                                  indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def recipe():
    return json.loads((ROOT/'recipe.json').read_text(encoding='utf-8'))

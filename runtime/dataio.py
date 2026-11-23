"""Read exact indexed archive members; no spatial preprocessing."""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections import OrderedDict
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from common import MARKERS, ORGANS


def member_role(member):
    path = PurePosixPath(member)
    if path.is_absolute() or '..' in path.parts or '\\' in member:
        raise ValueError('Unsafe archive member')
    for role in ('train', 'test'):
        if role in path.parts:
            return role
    raise ValueError('Missing official train/test directory')


def load_index(path, marker=None, subset=None):
    with open(path, encoding='utf-8', newline='') as stream:
        source = list(csv.DictReader(stream))
    selected = []
    seen = set()
    for row in source:
        if row['organ'] not in ORGANS or row['subset'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown indexed organ/subset')
        if marker is not None and row.get('marker') != marker:
            continue
        if subset is not None and row['subset'] != subset:
            continue
        role = 'test' if row['subset'] == 'test' else 'train'
        if member_role(row['dapi']) != role or '/DAPI/' not in row['dapi']:
            raise ValueError('DAPI split/channel mismatch')
        if row['subset'] != 'test':
            if row.get('marker') not in MARKERS:
                raise ValueError('Unknown training marker')
            if member_role(row['target']) != 'train':
                raise ValueError('Target is not official Train')
            if '/' + row['marker'] + '/' not in row['target']:
                raise ValueError('Target marker mismatch')
        key = (row['organ'], row.get('marker', ''), row['stem'])
        if key in seen:
            raise ValueError('Duplicate sample in index')
        seen.add(key)
        selected.append(row)
    if not selected:
        raise ValueError('No indexed images selected')
    return selected


class ArchiveReader:
    def __init__(self, path, cache_size=256):
        self.archive = zipfile.ZipFile(path)
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.archive.close()

    def image(self, member, expected_sha=None):
        member_role(member)
        key = (member, expected_sha)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        content = self.archive.read(member)
        if expected_sha and hashlib.sha256(content).hexdigest() != expected_sha:
            raise ValueError('Image SHA256 mismatch: ' + member)
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.size != (256, 256) or image.mode != 'RGB':
                raise ValueError('Expected original RGB 256x256 image: ' + member)
            value = np.asarray(image.convert('L'), dtype=np.float32)[None]/255.0
        value.setflags(write=False)
        self.cache[key] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def pair(self, row):
        if row['subset'] not in ('train', 'val'):
            raise ValueError('Pair loading only accepts official Train-derived rows')
        if member_role(row['dapi']) != 'train' or member_role(row['target']) != 'train':
            raise ValueError('Test member cannot be used for training or validation')
        return (self.image(row['dapi'], row.get('dapi_sha256')),
                self.image(row['target'], row.get('target_sha256')))

    def input(self, row):
        expected = 'test' if row['subset'] == 'test' else 'train'
        if member_role(row['dapi']) != expected:
            raise ValueError('Input split mismatch')
        return self.image(row['dapi'], row.get('dapi_sha256'))


def paired_augmentation(x, y, step, position, stem, seed=2026):
    value = hashlib.sha256(('%d|%d|%d|%s' % (seed, step, position, stem)).encode()).digest()[0] % 8
    def transform(image):
        result = np.rot90(image, value % 4, axes=(-2, -1))
        if value // 4:
            result = np.flip(result, axis=-1)
        return np.array(result, copy=True, order='C')
    return transform(x), transform(y)

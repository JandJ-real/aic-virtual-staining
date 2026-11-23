"""Only indexed Test DAPI enters the model; no target loading or transforms."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

from contract import TASKS, ZIP_NAME

EXPECTED_COUNTS = dict(colon=1346, liver=560, stomach=280)


def validate_inputs(rows, require_full=True):
    seen = set()
    for r in rows:
        path = PurePosixPath(r['dapi'])
        key = (r['organ'], r['stem'])
        if (r['organ'] not in TASKS or r['subset'] != 'test' or r.get('marker')
                or r.get('target') or r.get('target_sha256') or key in seen
                or path.is_absolute() or '..' in path.parts or '\\' in r['dapi']
                or path.parts[-3:-1] != ('test', 'DAPI') or path.stem != r['stem']
                or not re.fullmatch(r'ROI\d+_\d+_\d+', r['stem'])
                or not re.fullmatch(r'[a-f0-9]{64}', r['dapi_sha256'])):
            raise ValueError('Invalid/duplicate Test input or unexpected target/Marker')
        seen.add(key)
    if not rows or (require_full and Counter(r['organ'] for r in rows) != EXPECTED_COUNTS):
        raise ValueError('Incomplete frozen Test DAPI list')
    return rows


def load_inputs(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return validate_inputs(list(csv.DictReader(stream)))


def verify_official_inputs(rows, archive):
    OFFICIAL_DIRS={'colon':'colon（包含训练集和测试集输入）','liver':'liver(包括训练集和测试集输入)','stomach':'stomach（包含训练集和测试集输入）'}
    validate_inputs(rows)
    wanted = {r['dapi'] for r in rows}
    with zipfile.ZipFile(archive) as z:
        members = [p for p in z.namelist() if '/test/DAPI/' in p and not p.endswith('/')]
        if len(members) != len(set(members)) or set(members) != wanted:
            raise ValueError('Official Test input list differs from frozen manifest')
        for r in rows:
            if PurePosixPath(r['dapi']).parts[-4] != OFFICIAL_DIRS[r['organ']]:
                raise ValueError('Organ input identity mismatch')
            if hashlib.sha256(z.read(r['dapi'])).hexdigest() != r['dapi_sha256']:
                raise ValueError('Official Test DAPI bytes changed')
    return dict(images=len(rows), organs=dict(EXPECTED_COUNTS), test_gt_read=False,
                model_inference_performed=False)


def output_name(row):
    return 'results/%s/test/%s/%s_fake.png' % (row['organ'], TASKS[row['organ']], row['stem'])


def verify_zip(path, rows):
    from PIL import Image
    validate_inputs(rows, require_full=False)
    path = Path(path)
    if path.stat().st_size > 2048*1024**2:
        raise ValueError('Compressed submission exceeds 2048 MiB')
    expected = {output_name(r) for r in rows}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(expected) or set(names) != expected or len(names) > 20000:
            raise ValueError('ZIP contains missing/duplicate/extra or incorrectly routed files')
        infos = z.infolist()
        if sum(i.file_size for i in infos) > 8192*1024**2 or any(i.file_size > 200*1024**2 for i in infos):
            raise ValueError('Uncompressed official limits exceeded')
        for name in names:
            with Image.open(io.BytesIO(z.read(name))) as image:
                image.load()
                if image.format != 'PNG' or image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError('Expected native 256x256 RGB PNG: '+name)
    return dict(images=len(rows), organs=dict(Counter(r['organ'] for r in rows)),
                zip_bytes=path.stat().st_size, layout_checked=True, all_png_checked=True)


def predict(network, reader, rows, device, output, progress=None, final_guard=None):
    import numpy as np
    from engine import evaluation_context, single_prediction
    from metrics import png_pixels
    validate_inputs(rows, require_full=False)
    output = Path(output)
    partial = output.with_name(output.name+'.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Existing output/partial; automatic repetition is forbidden')
    output.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_context(network):
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for n, row in enumerate(rows, 1):
                x = reader.input(row)
                if x.dtype != np.float32 or x.shape != (1, 256, 256) or not np.isfinite(x).all():
                    raise ValueError('Original single FP32 DAPI256 required')
                task = dict(row, marker=TASKS[row['organ']])
                prediction = single_prediction(network, x, device, row=task)
                if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
                    raise ValueError('Nonfinite/out-of-range model output; no repair permitted')
                encoded, _ = png_pixels(prediction)
                archive.writestr(output_name(row), encoded)
                if progress and (n == 1 or n % 50 == 0 or n == len(rows)):
                    progress(n)
    result = verify_zip(partial, rows)
    if final_guard:
        final_guard()
    # Atomic no-overwrite publication, on the same filesystem.
    os.link(partial, output)
    partial.unlink()
    return result

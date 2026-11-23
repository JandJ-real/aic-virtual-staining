"""Pinned rematch Gaussian11 algorithm, semifinal clipping and PNG pixels."""
from __future__ import annotations

import io
import math
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from common import ORGANS, WEIGHTS


def png_pixels(prediction):
    array = np.asarray(prediction)
    if array.shape != (256, 256) or not np.isfinite(array).all():
        raise ValueError('Expected finite single-channel 256x256 prediction')
    gray = np.rint(np.clip(array, 0, 1)*255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    stream = io.BytesIO()
    Image.fromarray(rgb, mode='RGB').save(stream, format='PNG', compress_level=6)
    encoded = stream.getvalue()
    with Image.open(io.BytesIO(encoded)) as image:
        decoded = np.array(image, dtype=np.uint8)
    return encoded, decoded


def image_metrics(target_rgb, prediction_rgb):
    if (target_rgb.shape != (256, 256, 3) or prediction_rgb.shape != target_rgb.shape
            or target_rgb.dtype != np.uint8 or prediction_rgb.dtype != np.uint8):
        raise ValueError('Scoring requires matching uint8 RGB256 images')
    # Explicit independent 2D channels preserve RGB mean SSIM on old and new APIs.
    raw = float(np.mean([structural_similarity(target_rgb[..., channel], prediction_rgb[..., channel],
        data_range=255, win_size=11, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False) for channel in range(3)]))
    error = prediction_rgb.astype(np.float64)-target_rgb.astype(np.float64)
    mse = float(np.square(error).mean())
    psnr = float('inf') if mse == 0 else 10*math.log10(255**2/mse)
    ssim = min(max(raw, 0.0), 1.0)
    normalized = min(max(psnr/50, 0.0), 1.0)
    return {'ssim': ssim, 'raw_ssim': raw, 'psnr': psnr,
            'normalized_psnr': normalized, 'score': 100*(0.7*ssim+0.3*normalized),
            'mae': float(np.abs(error).mean()), 'bias': float(error.mean())}


def average(records):
    result = {'images': len(records)}
    for key in ('score', 'ssim', 'psnr', 'mae', 'bias'):
        values = [float(row[key]) for row in records if key in row]
        if values:
            result[key] = sum(values)/len(values)
    result['psnr_infinite_count'] = sum(math.isinf(float(r['psnr'])) for r in records)
    return result


def aggregate(records):
    groups = defaultdict(list)
    rois = defaultdict(list)
    for row in records:
        groups[row['organ']].append(row)
        rois[(row['organ'], row['roi'])].append(row)
    if set(groups) != set(ORGANS):
        raise ValueError('All three organs must be evaluated')
    organs = {organ: average(groups[organ]) for organ in ORGANS}
    result = {'score': sum(WEIGHTS[o]*organs[o]['score'] for o in ORGANS),
            'organs': organs,
            'rois': {o + '/' + roi: average(rows) for (o, roi), rows in sorted(rois.items())}}
    if all('target_mean' in row for row in records):
        def strata(rows):
            return {name: average(part) if part else {'images': 0} for name, part in (
                ('target_mean_le4', [r for r in rows if r['target_mean'] <= 4]),
                ('target_mean_gt4', [r for r in rows if r['target_mean'] > 4]))}
        result['signal_strata'] = {organ: strata(groups[organ]) for organ in ORGANS}
        result['roi_signal_strata'] = {o+'/'+roi: strata(rows) for (o, roi), rows in sorted(rois.items())}
        result['diagnostic_threshold'] = 'GT image mean <=4 on uint8 pixels; all images remain in score'
    return result

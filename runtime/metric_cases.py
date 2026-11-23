"""Small deterministic synthetic fixtures for runtime numeric contract checks."""
import numpy as np


def cases():
    y, x = np.indices((256, 256), dtype=np.int32)
    def rgb(gray):
        return np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=-1)
    black = np.zeros((256, 256, 3), dtype=np.uint8)
    ramp = np.stack((x, y, (3*x+5*y) % 256), axis=-1).astype(np.uint8)
    checker = rgb(((x//8+y//8) % 2)*255)
    sparse = rgb(np.where(((x % 29)-14)**2+((y % 31)-15)**2 < 20, 200, 2))
    edges = rgb(np.where((x < 3) | (x > 252) | (y < 3) | (y > 252), 255, 0))
    return [
        ('perfect_black', black, black.copy()),
        ('black_white', black, np.full_like(black, 255)),
        ('near_black', np.full_like(black, 1), black),
        ('constant_64_65', np.full_like(black, 64), np.full_like(black, 65)),
        ('rgb_ramp_shift', ramp, np.roll(ramp, 3, axis=1)),
        ('rgb_channel_permutation', ramp, ramp[..., [2, 0, 1]]),
        ('edge_strip', edges, np.roll(edges, 2, axis=0)),
        ('sparse_structures', sparse, np.roll(sparse, 1, axis=1)),
        ('negative_ssim', checker, 255-checker),
    ]

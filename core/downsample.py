"""
downsample.py
-------------
Decimation: pick every Nth pixel after low-pass filtering.

This is where the actual "compression" (dimension reduction) happens.
The filtering step (filters.py) must run BEFORE this, or you're violating
Nyquist and will see aliasing artifacts -- that comparison (filtered vs
unfiltered downsampling) is your key demo.
"""

import numpy as np


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Keep every `factor`-th pixel along both spatial axes.
    factor=2 -> half width, half height -> 1/4 the pixel count.
    factor=4 -> quarter width, quarter height -> 1/16 the pixel count.

    image: (H, W) or (H, W, C)
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    return image[::factor, ::factor, ...]


def compression_ratio(original_shape, downsampled_shape) -> float:
    """
    Ratio of pixel counts (a simple, honest stand-in for storage size,
    since pixel count directly determines the size of what you'd save
    to disk before any further entropy coding).
    """
    orig_pixels = np.prod(original_shape[:2])
    down_pixels = np.prod(downsampled_shape[:2])
    return orig_pixels / down_pixels

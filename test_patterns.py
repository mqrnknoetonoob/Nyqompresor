"""
test_patterns.py
-----------------
Synthetic images specifically designed to show aliasing clearly.
Real photos often don't show aliasing dramatically -- these patterns will.
Use these for your "filter vs no filter" comparison figure in the report.
"""

import numpy as np


def checkerboard(size: int = 512, square: int = 4) -> np.ndarray:
    """
    Fine checkerboard -- very high spatial frequency. Downsampling this
    without a low-pass filter produces dramatic moire/aliasing patterns.
    """
    row = (np.arange(size) // square) % 2
    col = (np.arange(size) // square) % 2
    pattern = np.logical_xor(row[:, None], col[None, :]).astype(np.uint8) * 255
    return pattern


def sine_grating(size: int = 512, frequency: float = 0.4) -> np.ndarray:
    """
    Sine wave grating with increasing frequency left-to-right -- a classic
    test signal for visualizing exactly where aliasing kicks in relative
    to the sampling rate.
    frequency: cycles per pixel at the right edge (higher = tighter lines).
    """
    x = np.linspace(0, 1, size)
    freq_ramp = np.linspace(0, frequency, size)
    row = 127 + 127 * np.sin(2 * np.pi * freq_ramp * x * size)
    pattern = np.tile(row, (size, 1)).astype(np.uint8)
    return pattern


def radial_zone_plate(size: int = 512) -> np.ndarray:
    """
    Zone plate: frequency increases with distance from center, radially.
    Classic image-processing test pattern for visualizing aliasing in 2D
    (produces concentric ring artifacts when downsampled without filtering).
    """
    y, x = np.meshgrid(np.arange(size) - size // 2, np.arange(size) - size // 2, indexing="ij")
    r = np.sqrt(x**2 + y**2)
    pattern = 127 + 127 * np.cos((r ** 2) * (np.pi / (size * 8)))
    return pattern.astype(np.uint8)

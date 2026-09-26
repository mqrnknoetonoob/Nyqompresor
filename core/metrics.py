"""
metrics.py
----------
Quantitative comparison between original and reconstructed images.
This is what turns "it looks blurry" into a number you can plot and defend.
"""

import numpy as np


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error. Lower = closer to original."""
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    return float(np.mean((original - reconstructed) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio, in dB. Higher = better quality.
    Standard formula: PSNR = 10 * log10(MAX^2 / MSE)
    Typical references: >40dB excellent, 30-40dB good, <20dB poor.
    """
    error = mse(original, reconstructed)
    if error == 0:
        return float("inf")  # identical images
    return 10 * np.log10((max_pixel ** 2) / error)


def compression_ratio(original_shape, stored_shape) -> float:
    """Ratio of pixel counts, original vs stored (downsampled) size."""
    orig_pixels = np.prod(original_shape[:2])
    stored_pixels = np.prod(stored_shape[:2])
    return float(orig_pixels / stored_pixels)


def summarize(original: np.ndarray, reconstructed: np.ndarray, stored_shape) -> dict:
    return {
        "mse": mse(original, reconstructed),
        "psnr_db": psnr(original, reconstructed),
        "compression_ratio": compression_ratio(original.shape, stored_shape),
    }

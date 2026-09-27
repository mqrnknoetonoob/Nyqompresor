"""
metrics.py
----------
Quantitative comparison between original and reconstructed images.
This is what turns "it looks blurry" into a number you can plot and defend.

Two families of metrics are provided:

  Pixel-count metrics (compression_ratio, chroma_compression_ratio):
    Ratio of spatial dimensions only — does not account for actual byte savings.

  Byte-level metrics (compressed_bytes, full_summary, chroma_full_summary):
    Compress the raw stored array with zlib (deflate) and measure the result.
    This is a realistic lower-bound on the file size that a real encoder would
    produce after entropy coding the downsampled data, and is directly
    comparable across all three pipeline methods.
"""

import zlib
import numpy as np


# ------------------------------------------------------------------ #
#  Signal quality                                                      #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Pixel-count compression ratios (spatial only)                      #
# ------------------------------------------------------------------ #

def compression_ratio(original_shape, stored_shape) -> float:
    """Ratio of pixel counts, original vs stored (downsampled) size."""
    orig_pixels = np.prod(original_shape[:2])
    stored_pixels = np.prod(stored_shape[:2])
    return float(orig_pixels / stored_pixels)


def chroma_compression_ratio(h: int, w: int, luma_factor: int, chroma_factor: int) -> float:
    """
    True stored-pixel compression ratio for YCbCr chroma subsampling.

    Unlike the uniform pipeline where all 3 channels are downsampled equally,
    chroma subsampling stores the Y channel at luma_factor resolution and the
    Cb, Cr channels at chroma_factor resolution independently. This computes
    the honest ratio of total original channel pixels to total stored channel
    pixels across all three planes.

    Formula:
        original  = H * W * 3                        (3 full-res channels)
        stored    = (H/luma)*(W/luma)                (Y plane)
                  + 2*(H/chroma)*(W/chroma)          (Cb + Cr planes)
        CR = original / stored
    """
    original_pixels = h * w * 3
    stored_pixels = (
        (h // luma_factor) * (w // luma_factor)
        + 2 * (h // chroma_factor) * (w // chroma_factor)
    )
    if stored_pixels == 0:
        return float("inf")
    return float(original_pixels / stored_pixels)


# ------------------------------------------------------------------ #
#  Byte-level compression (zlib / deflate entropy coding)             #
# ------------------------------------------------------------------ #

def compressed_bytes(arr: np.ndarray, level: int = 6) -> int:
    """
    Compress a uint8 numpy array with zlib (deflate, level 6 by default)
    and return the resulting byte count.

    This simulates what an entropy coder (RLE + Huffman, or deflate) would
    produce when writing the downsampled pixel data to disk. It is the
    standard baseline used to demonstrate that spatial decimation DOES
    produce real byte savings once entropy coding is applied.

    level=6 is zlib's default balance of speed vs compression.
    """
    raw = arr.astype(np.uint8).tobytes()
    return len(zlib.compress(raw, level=level))


def byte_compression_ratio(original: np.ndarray, stored: np.ndarray) -> float:
    """
    Ratio of the zlib-compressed original to the zlib-compressed stored
    (downsampled) array. Both sides are compressed so the comparison is
    fair — we are measuring the net savings from dimension reduction, not
    from the compressor alone.
    """
    orig_b  = compressed_bytes(original)
    stored_b = compressed_bytes(stored)
    return float(orig_b / stored_b) if stored_b > 0 else float("inf")


# ------------------------------------------------------------------ #
#  Convenience summary dicts (one per pipeline type)                  #
# ------------------------------------------------------------------ #

def summarize(original: np.ndarray, reconstructed: np.ndarray, stored_shape) -> dict:
    """
    Legacy pixel-count summary — kept for backward compatibility with
    test_pipeline.py.  New code should call full_summary instead.
    """
    return {
        "mse": mse(original, reconstructed),
        "psnr_db": psnr(original, reconstructed),
        "compression_ratio": compression_ratio(original.shape, stored_shape),
    }


def full_summary(original: np.ndarray, reconstructed: np.ndarray,
                  stored_arr: np.ndarray) -> dict:
    """
    Full metrics for the Nyquist-Shannon custom pipeline and the OpenCV
    baseline: pixel-count CR plus zlib byte-level CR and raw byte counts.

    stored_arr: the actual downsampled uint8 array (not just its shape),
                needed to compute real byte sizes.
    """
    orig_b   = compressed_bytes(original)
    stored_b = compressed_bytes(stored_arr)
    return {
        "mse":                    mse(original, reconstructed),
        "psnr_db":                psnr(original, reconstructed),
        "compression_ratio":      compression_ratio(original.shape, stored_arr.shape),
        "original_bytes":         orig_b,
        "stored_bytes":           stored_b,
        "byte_compression_ratio": float(orig_b / stored_b) if stored_b > 0 else float("inf"),
    }


def chroma_full_summary(original: np.ndarray, reconstructed: np.ndarray,
                         downsampled_planes: list,
                         luma_factor: int, chroma_factor: int) -> dict:
    """
    Full metrics for the YCbCr chroma subsampling pipeline.

    downsampled_planes: list of three 2D uint8 arrays [Y_small, Cr_small, Cb_small]
                        as returned by color.chroma_subsample_pipeline.
    The stored byte size is the sum of the zlib-compressed sizes of all three
    planes — mirroring how a real codec would store them separately.
    """
    h, w = original.shape[:2]
    orig_b   = compressed_bytes(original)
    stored_b = sum(compressed_bytes(p) for p in downsampled_planes)
    return {
        "mse":                    mse(original, reconstructed),
        "psnr_db":                psnr(original, reconstructed),
        "compression_ratio":      chroma_compression_ratio(h, w, luma_factor, chroma_factor),
        "original_bytes":         orig_b,
        "stored_bytes":           stored_b,
        "byte_compression_ratio": float(orig_b / stored_b) if stored_b > 0 else float("inf"),
    }

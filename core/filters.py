"""
filters.py
----------
Low-pass spatial filtering via 2D convolution.

THEORY RECAP (for your report):
An image downsampled naively can alias: high-frequency detail (fine texture,
sharp edges) gets misrepresented as false low-frequency patterns (moire,
jagged edges) once you throw away samples. The Nyquist-Shannon theorem says
you must not have signal content above half the new sampling rate once you
resample. A low-pass (blur) filter removes that high-frequency content
BEFORE downsampling, so there's nothing left to alias.

We implement 2D convolution ourselves (no cv2.filter2D, no scipy.ndimage) so
you can point to this code and say "this is literally what removes the
frequencies that would alias."
"""

import numpy as np


def box_kernel(size: int) -> np.ndarray:
    """
    Simplest possible low-pass filter: an averaging window.
    size=3 -> every pixel becomes the average of its 3x3 neighborhood.
    Larger size = stronger blur = more aggressive high-frequency removal.
    """
    if size % 2 == 0:
        raise ValueError("kernel size should be odd (so it has a center pixel)")
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian blur kernel. Weights fall off smoothly from the center,
    which gives a cleaner frequency-domain cutoff than the box filter
    (box filter has sharp edges in space -> ripples in frequency domain;
    not something you need to derive, just worth a one-line mention in
    your report if you compare the two).
    """
    if size % 2 == 0:
        raise ValueError("kernel size should be odd")
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2D convolution (technically cross-correlation, which is what
    every image-processing library actually does under the name
    "convolution" -- kernels here are symmetric so it doesn't matter).

    channel: 2D array (single color channel), float64
    kernel:  2D array, odd size, sums to 1 (a low-pass filter)

    Implementation note: a naive 4-nested-for-loop convolution in pure
    Python is extremely slow for real images (millions of pixels x kernel
    area). We use numpy's stride tricks to build a "sliding window" view
    of the padded image without copying data, then do the weighted sum
    with vectorized numpy ops. This is still literally sliding the kernel
    over every pixel -- just done fast. If you want to see the "obvious"
    slow version for your report/understanding, see convolve2d_naive below.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Reflect padding avoids darkened/artifact edges you'd get from
    # zero-padding (which pretends the image is surrounded by black).
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    out_h, out_w = channel.shape
    # Build a view of shape (out_h, out_w, kh, kw): for every output pixel,
    # the kh x kw neighborhood that produces it.
    shape = (out_h, out_w, kh, kw)
    strides = padded.strides[:2] * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Weighted sum of each window with the kernel = convolution result.
    result = np.einsum("ijkl,kl->ij", windows, kernel)
    return result


def convolve2d_naive(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Slow, fully explicit reference implementation. Use this on a small
    test image if you want to demonstrate/understand the raw sliding-
    window mechanics step by step (e.g. in a notebook), not on full images.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out_h, out_w = channel.shape
    result = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            region = padded[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)

    return result


def low_pass_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0,
                     kernel_type: str = "gaussian") -> np.ndarray:
    """
    Apply the low-pass filter to an image. Works for both grayscale
    (H, W) and color (H, W, C) arrays -- each channel filtered independently
    since each channel is its own 2D spatial signal.
    """
    if kernel_type == "box":
        kernel = box_kernel(kernel_size)
    elif kernel_type == "gaussian":
        kernel = gaussian_kernel(kernel_size, sigma)
    else:
        raise ValueError("kernel_type must be 'box' or 'gaussian'")

    image = image.astype(np.float64)

    if image.ndim == 2:
        return convolve2d(image, kernel)

    if image.ndim == 3:
        channels = [convolve2d(image[:, :, c], kernel) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)

    raise ValueError("image must be 2D (grayscale) or 3D (H, W, C)")

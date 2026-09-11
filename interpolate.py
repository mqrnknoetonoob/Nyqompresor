"""
interpolate.py
--------------
Reconstruction: rebuild a full-size image from the downsampled grid.

Two modes, as in your proposal:
  - ZOH (Zero Order Hold) = nearest-neighbor: each new pixel just copies
    the nearest known sample. Cheap, blocky, no smoothing -- good baseline
    to show *why* interpolation matters.
  - Linear (bilinear in 2D) = linear interpolation along rows, then columns.
    Smoother, closer to the true signal between samples.

Both implemented from scratch (no cv2.resize / scipy.ndimage.zoom).
"""

import numpy as np


def upsample_nearest(small: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Zero Order Hold reconstruction. For each output pixel, find the closest
    source pixel (by proportional position) and copy its value directly.
    """
    src_h, src_w = small.shape[0], small.shape[1]

    # Map each output coordinate back into source coordinate space.
    row_idx = np.floor(np.arange(out_h) * (src_h / out_h)).astype(int)
    col_idx = np.floor(np.arange(out_w) * (src_w / out_w)).astype(int)
    row_idx = np.clip(row_idx, 0, src_h - 1)
    col_idx = np.clip(col_idx, 0, src_w - 1)

    return small[row_idx][:, col_idx, ...]


def _linear_interp_1d(values: np.ndarray, out_len: int) -> np.ndarray:
    """
    Linear interpolation along a single axis (used twice by
    upsample_bilinear: once for rows, once for columns).

    values: 1D array of samples
    out_len: desired output length
    """
    src_len = len(values)
    if src_len == 1:
        return np.repeat(values, out_len)

    # Position of each output sample in source-index space (fractional).
    src_positions = np.linspace(0, src_len - 1, out_len)

    left_idx = np.floor(src_positions).astype(int)
    right_idx = np.clip(left_idx + 1, 0, src_len - 1)
    frac = src_positions - left_idx  # fractional distance to the right sample

    # Linear interpolation formula: v = v_left*(1-frac) + v_right*frac
    return values[left_idx] * (1 - frac) + values[right_idx] * frac


def upsample_bilinear(small: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Bilinear reconstruction: linear interpolation along rows, then along
    columns. Works for grayscale (H, W) and color (H, W, C).
    """
    src_h, src_w = small.shape[0], small.shape[1]

    # Step 1: interpolate along columns (width) for every existing row.
    stage1 = np.zeros((src_h, out_w) + small.shape[2:], dtype=np.float64)
    for i in range(src_h):
        stage1[i, ...] = _apply_1d_along_axis(small[i, ...], out_w)

    # Step 2: interpolate along rows (height) for every new column.
    stage2 = np.zeros((out_h, out_w) + small.shape[2:], dtype=np.float64)
    for j in range(out_w):
        stage2[:, j, ...] = _apply_1d_along_axis(stage1[:, j, ...], out_h)

    return stage2


def _apply_1d_along_axis(values: np.ndarray, out_len: int) -> np.ndarray:
    """
    Helper: values may be (src_len,) for grayscale or (src_len, C) for
    color. Interpolate each channel independently if color.
    """
    if values.ndim == 1:
        return _linear_interp_1d(values, out_len)
    # (src_len, C) -> interpolate each channel
    out = np.stack(
        [_linear_interp_1d(values[:, c], out_len) for c in range(values.shape[1])],
        axis=-1,
    )
    return out


def reconstruct(small: np.ndarray, out_h: int, out_w: int, mode: str = "linear") -> np.ndarray:
    if mode == "zoh" or mode == "nearest":
        return upsample_nearest(small, out_h, out_w)
    elif mode == "linear" or mode == "bilinear":
        return upsample_bilinear(small, out_h, out_w)
    else:
        raise ValueError("mode must be 'zoh' or 'linear'")

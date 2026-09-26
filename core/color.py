"""
color.py
--------
RGB is just three stacked 2D spatial signals (R, G, B channels), so the
core pipeline in pipeline.py already handles color images directly --
filters.low_pass_filter, downsample, and interpolate.reconstruct all
branch on image.ndim and process each channel independently.

This module adds the OPTIONAL stretch goal: YCbCr chroma subsampling,
which downsamples color information (Cb, Cr) more aggressively than
brightness (Y), since human vision is far more sensitive to luminance
than chrominance. This is the same principle real codecs (JPEG, video)
use, and is a nice extra comparison point for your report.
"""

import numpy as np
import cv2
from . import pipeline as pl


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2YCrCb)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def chroma_subsample_pipeline(image: np.ndarray, luma_factor: int, chroma_factor: int,
                               recon_mode: str = "linear", kernel_size: int = 5,
                               sigma: float = 1.0) -> dict:
    """
    Downsample luma (Y) lightly (or not at all) and chroma (Cb, Cr) more
    aggressively, then reconstruct each independently and recombine.

    luma_factor=1 means no downsampling of brightness at all.
    chroma_factor > luma_factor is the typical real-world choice.

    Returns a dict:
        "reconstructed"     : uint8 RGB array at the original resolution.
        "downsampled_planes": list of three 2D uint8 arrays [Y_small, Cr_small, Cb_small].
                              These are what would actually be written to disk — the three
                              planes at their reduced resolutions.  Needed by
                              metrics.chroma_full_summary to compute honest byte sizes.
    """
    ycrcb = rgb_to_ycbcr(image)  # cv2 gives Y, Cr, Cb order
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

    def process_channel(channel_2d, factor):
        """Run the full encode-decode pipeline on a single 2D channel.
        Returns (reconstructed_2d, downsampled_2d) both as uint8."""
        channel_3d = channel_2d[:, :, np.newaxis]  # reuse pipeline's (H,W,C) path
        result = pl.run_pipeline(
            channel_3d, factor, recon_mode=recon_mode,
            kernel_size=kernel_size, sigma=sigma, apply_filter=(factor > 1),
        )
        recon      = result["reconstructed"][:, :, 0]   # 2D uint8 reconstructed
        downsampled = result["downsampled"][:, :, 0]    # 2D uint8 stored plane
        return recon, downsampled

    y_out,  y_small  = process_channel(y,  luma_factor)
    cr_out, cr_small = process_channel(cr, chroma_factor)
    cb_out, cb_small = process_channel(cb, chroma_factor)

    merged = np.stack([y_out, cr_out, cb_out], axis=-1)

    return {
        "reconstructed":      ycbcr_to_rgb(merged),
        "downsampled_planes": [y_small, cr_small, cb_small],
    }

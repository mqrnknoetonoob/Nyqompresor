"""
pipeline.py
-----------
Your from-scratch compressor: chains filter -> downsample -> reconstruct.

This is the "encoder + decoder" your proposal describes as steps 1-4.
"""

import numpy as np
from . import filters, downsample as ds, interpolate as interp


def should_compress(image: np.ndarray, mode: str = "always") -> bool:
    """
    Preprocessing gate (see project notes: this is a deliberate extension
    point for a future "detect sensitive documents" classifier -- NOT
    implemented in this version, always returns True for mode='always').
    """
    if mode == "always":
        return True
    raise NotImplementedError(
        "Sensitive-document detection is a planned future extension, not implemented yet."
    )


def encode(image: np.ndarray, factor: int, kernel_size: int = 5,
           sigma: float = 1.0, kernel_type: str = "gaussian",
           apply_filter: bool = True) -> np.ndarray:
    """
    Compress: (optionally) anti-alias filter, then downsample.

    apply_filter=False lets you deliberately skip the anti-aliasing step
    to demonstrate aliasing artifacts -- keep this for your report demo.
    """
    working = image.astype(np.float64)

    if apply_filter:
        working = filters.low_pass_filter(
            working, kernel_size=kernel_size, sigma=sigma, kernel_type=kernel_type
        )

    downsampled = ds.downsample(working, factor)
    return np.clip(downsampled, 0, 255)


def decode(downsampled: np.ndarray, out_h: int, out_w: int, mode: str = "linear") -> np.ndarray:
    """Reconstruct to original dimensions."""
    reconstructed = interp.reconstruct(downsampled, out_h, out_w, mode=mode)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def run_pipeline(image: np.ndarray, factor: int, recon_mode: str = "linear",
                  kernel_size: int = 5, sigma: float = 1.0,
                  kernel_type: str = "gaussian", apply_filter: bool = True) -> dict:
    """
    Full round trip: encode then decode, returning intermediate results
    so you can inspect/display each stage.
    """
    out_h, out_w = image.shape[0], image.shape[1]

    downsampled = encode(
        image, factor, kernel_size=kernel_size, sigma=sigma,
        kernel_type=kernel_type, apply_filter=apply_filter,
    )
    downsampled_uint8 = downsampled.astype(np.uint8)

    reconstructed = decode(downsampled, out_h, out_w, mode=recon_mode)

    return {
        "downsampled": downsampled_uint8,
        "reconstructed": reconstructed,
    }

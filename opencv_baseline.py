"""
opencv_baseline.py
-------------------
The "black box" comparison method. Thin wrapper around cv2.resize.

Kept deliberately separate from your own pipeline (core/pipeline.py) so
the comparison is honest: this file contains none of your own filter/
downsample/interpolate code.

NOTE on fairness (see project discussion): cv2.resize with INTER_AREA is
generally considered OpenCV's best option for shrinking (it does area-
based averaging, which acts somewhat like a low-pass filter). Plain
INTER_LINEAR when shrinking does NOT properly anti-alias. We expose both
so you can choose what to compare against and discuss the difference in
your report.
"""

import numpy as np
import cv2


def opencv_round_trip(image: np.ndarray, factor: int, down_interp: str = "area",
                       up_interp: str = "linear") -> dict:
    """
    Downsample then upsample using OpenCV, mirroring your own pipeline's
    round trip so metrics are directly comparable.
    """
    h, w = image.shape[0], image.shape[1]
    small_h, small_w = max(1, h // factor), max(1, w // factor)

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
    }

    img = image.astype(np.uint8)

    downsampled = cv2.resize(img, (small_w, small_h), interpolation=interp_map[down_interp])
    reconstructed = cv2.resize(downsampled, (w, h), interpolation=interp_map[up_interp])

    return {
        "downsampled": downsampled,
        "reconstructed": reconstructed,
    }

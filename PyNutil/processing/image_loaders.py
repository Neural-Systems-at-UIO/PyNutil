from __future__ import annotations

import numpy as np


def detect_pixel_id(segmentation: np.ndarray) -> np.ndarray:
    """Infer the foreground pixel id from the first non-background region."""

    if segmentation.ndim == 2:
        non_zero = segmentation[segmentation != 0]
        if non_zero.size > 0:
            pixel_id = [int(non_zero[0])]
        else:
            pixel_id = [255]
    else:
        mask = ~np.all(segmentation == 0, axis=2)
        segmentation_no_background = segmentation[mask]
        if segmentation_no_background.size > 0:
            pixel_id = segmentation_no_background[0]
        else:
            pixel_id = [255, 255, 255]
    return np.asarray(pixel_id)


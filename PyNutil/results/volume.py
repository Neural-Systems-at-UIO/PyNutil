from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VolumeResult:
    """Volumes returned by :func:`~PyNutil.interpolate_volume`.

    Attributes
    ----------
    value:
        Atlas-space value volume (float32). Each voxel holds the accumulated
        metric (pixel count, mean intensity, or object count) for that
        location. ``None`` when the caller did not request this output.
    frequency:
        Per-voxel sample-count volume (uint32). Records how many section
        pixels contributed to each atlas voxel. ``None`` when not requested.
    damage:
        Binary damage-mask volume (uint8). A value of 1 indicates that the
        corresponding atlas voxel overlaps a damaged region. ``None`` when
        not requested.
    """

    value: Optional[np.ndarray]
    frequency: Optional[np.ndarray]
    damage: Optional[np.ndarray]

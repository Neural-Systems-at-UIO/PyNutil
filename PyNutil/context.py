"""Immutable context objects for pipeline orchestration.

These frozen dataclasses carry state through the batch and section
processing layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .processing.adapters.base import SliceInfo
from .processing.adapters.segmentation import SegmentationAdapter


@dataclass(frozen=True)
class PipelineContext:
    """Immutable state shared across all sections in a batch run.

    Attributes
    ----------
    atlas_labels : pd.DataFrame
        Region lookup table (idx, name, r, g, b …).
    atlas_volume : np.ndarray or None
        3-D annotation volume for atlas-based slicing.
    hemi_map : np.ndarray or None
        3-D hemisphere mask (same shape as *atlas_volume*).
    segmentation_adapter : SegmentationAdapter
        Pre-resolved adapter for the current segmentation format.
    non_linear : bool
        Whether to apply non-linear deformation.
    object_cutoff : int
        Minimum connected-component area (binary pipeline).
    use_flat : bool
        If True, load flat-file atlas maps instead of slicing the volume.
    pixel_id : object
        Pixel colour to match, or ``"auto"`` for auto-detection.
    apply_damage_mask : bool
        Apply the damage / exclusion mask when available.
    intensity_channel : str or None
        Channel name for the intensity pipeline (``None`` in binary mode).
    min_intensity : int or None
        Lower intensity bound (intensity pipeline only).
    max_intensity : int or None
        Upper intensity bound (intensity pipeline only).
    """

    atlas_labels: pd.DataFrame
    atlas_volume: Optional[np.ndarray]
    hemi_map: Optional[np.ndarray]
    segmentation_adapter: SegmentationAdapter
    non_linear: bool
    object_cutoff: int
    use_flat: bool
    pixel_id: object
    apply_damage_mask: bool
    intensity_channel: Optional[str] = None
    min_intensity: Optional[int] = None
    max_intensity: Optional[int] = None


@dataclass(frozen=True)
class SectionContext:
    """Immutable per-section state built inside the batch loop.

    Attributes
    ----------
    section_number : int
        Numeric section identifier matching the alignment JSON.
    slice_info : SliceInfo
        Registration data for this section (anchoring, deformation, damage …).
    segmentation_path : str
        Path to the segmentation / image file on disk.
    flat_file_path : str or None
        Path to the corresponding flat-file atlas, if any.
    """

    section_number: int
    slice_info: SliceInfo
    segmentation_path: str
    flat_file_path: Optional[str] = None

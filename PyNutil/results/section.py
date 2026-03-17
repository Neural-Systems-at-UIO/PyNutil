from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SectionResult:
    """Result from processing a single section through atlas space."""

    points: np.ndarray
    centroids: np.ndarray
    region_areas: pd.DataFrame
    points_labels: np.ndarray
    centroids_labels: np.ndarray
    per_point_undamaged: np.ndarray
    per_centroid_undamaged: np.ndarray
    points_hemi_labels: np.ndarray
    centroids_hemi_labels: np.ndarray

    @classmethod
    def empty(cls, region_areas: pd.DataFrame | None = None) -> "SectionResult":
        """Create an empty result for skipped or empty sections."""
        return cls(
            points=np.array([], dtype=np.float64),
            centroids=np.array([], dtype=np.float64),
            region_areas=region_areas if region_areas is not None else pd.DataFrame(),
            points_labels=np.array([], dtype=np.int64),
            centroids_labels=np.array([], dtype=np.int64),
            per_point_undamaged=np.array([], dtype=bool),
            per_centroid_undamaged=np.array([], dtype=bool),
            points_hemi_labels=np.array([], dtype=np.int64),
            centroids_hemi_labels=np.array([], dtype=np.int64),
        )


@dataclass
class IntensitySectionResult:
    """Result from processing a single section in intensity mode."""

    region_intensities: Optional[pd.DataFrame]
    points: Optional[np.ndarray]
    points_labels: Optional[np.ndarray]
    points_hemi_labels: Optional[np.ndarray]
    point_intensities: Optional[np.ndarray]
    num_points: int = 0

    @classmethod
    def empty(cls) -> "IntensitySectionResult":
        """Create an empty result for skipped or empty sections."""
        return cls(
            region_intensities=None,
            points=None,
            points_labels=None,
            points_hemi_labels=None,
            point_intensities=None,
            num_points=0,
        )

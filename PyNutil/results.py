"""Data classes for PyNutil analysis results.

These dataclasses provide structured containers for the various results
produced during coordinate extraction, quantification, and volume interpolation.
They replace scattered instance attributes with clear, documented structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ── Single-section result (returned by section_processor) ───────────


@dataclass
class SectionResult:
    """Result from processing a single section through atlas space.

    Replaces the mutable-list-slot pattern where ``segmentation_to_atlas_space``
    wrote into nine pre-allocated shared lists by index.
    """

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
    def empty(cls, region_areas: pd.DataFrame | None = None) -> SectionResult:
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


# ── Per-section array bundles (used by quantification pipeline) ──────────


@dataclass
class PerEntityArrays:
    """Concatenated per-entity arrays across all sections.

    Groups the four parallel arrays that are always sliced / iterated
    together, replacing four positional parameters with one.
    Works for both pixel-points and object-centroids.
    """

    labels: np.ndarray
    """Atlas-region label for each entity."""
    hemi_labels: np.ndarray
    """Hemisphere label for each entity (1 = left, 2 = right, or ``None`` sentinel)."""
    undamaged: np.ndarray
    """Boolean flag: ``True`` if the entity is in an undamaged area."""
    section_lengths: List[int]
    """Number of entities contributed by each section (used for splitting)."""

    # ── convenience ──────────────────────────────────────────────────
    def split(self):
        """Yield per-section slices as ``(labels, hemi, undamaged)`` tuples.

        Uses offset-based slicing rather than ``np.split`` so that arrays
        whose total length exceeds ``sum(section_lengths)`` are handled
        gracefully (trailing elements are simply ignored).
        """
        offset = 0
        for n in self.section_lengths:
            lab = self.labels[offset : offset + n]
            hemi = self.hemi_labels[offset : offset + n]
            und = self.undamaged[offset : offset + n]
            yield lab, hemi, und
            offset += n


# ── Single-section intensity result (returned by intensity section processor) ──


@dataclass
class IntensitySectionResult:
    """Result from processing a single section in intensity mode.

    Replaces the mutable-list-slot pattern where
    ``segmentation_to_atlas_space_intensity`` wrote into six pre-allocated
    shared lists by index.
    """

    region_intensities: Optional[pd.DataFrame]
    """Per-region intensity DataFrame for this section."""
    points: Optional[np.ndarray]
    """3-D atlas-space coordinates of signal pixels (N×3)."""
    points_labels: Optional[np.ndarray]
    """Atlas-region label for each signal pixel."""
    points_hemi_labels: Optional[np.ndarray]
    """Hemisphere label for each signal pixel."""
    point_intensities: Optional[np.ndarray]
    """Intensity (or RGB) value for each signal pixel."""
    num_points: int = 0
    """Number of signal pixels in this section."""

    @classmethod
    def empty(cls) -> IntensitySectionResult:
        """Create an empty result for skipped or empty sections."""
        return cls(
            region_intensities=None,
            points=None,
            points_labels=None,
            points_hemi_labels=None,
            point_intensities=None,
            num_points=0,
        )


@dataclass
class AtlasData:
    """Bundle of atlas volume, hemisphere map, and region labels.

    Returned by :func:`load_atlas` for custom atlases.  Pipeline functions
    also accept a ``BrainGlobeAtlas`` instance directly.
    """

    volume: np.ndarray
    hemi_map: Optional[np.ndarray]
    labels: pd.DataFrame
    voxel_size_um: Optional[float] = None


@dataclass
class ExtractionResult:
    """Concatenated extraction output with context-aligned attribute names."""

    pixel_points: Optional[np.ndarray]
    centroids: Optional[np.ndarray]
    points_labels: Optional[np.ndarray]
    centroids_labels: Optional[np.ndarray]
    points_hemi_labels: Optional[np.ndarray]
    centroids_hemi_labels: Optional[np.ndarray]
    region_areas_list: List[pd.DataFrame]
    points_len: List[int]
    centroids_len: Optional[List[int]]
    segmentation_filenames: List[str]
    per_point_undamaged: Optional[np.ndarray]
    per_centroid_undamaged: Optional[np.ndarray]
    total_points_len: List[int]
    total_centroids_len: Optional[List[int]]
    region_intensities_list: Optional[List[Optional[pd.DataFrame]]] = None
    point_intensities: Optional[np.ndarray] = None
    # Custom-region mapped labels; populated by get_coordinates when custom regions are set
    points_custom_labels: Optional[np.ndarray] = None
    centroids_custom_labels: Optional[np.ndarray] = None

"""Data classes for PyNutil analysis results.

These dataclasses provide structured containers for the various results
produced during coordinate extraction, quantification, and volume interpolation.
They replace scattered instance attributes with clear, documented structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class CoordinateResults:
    """Results from coordinate extraction (get_coordinates).

    Contains transformed pixel/centroid coordinates and their atlas region assignments.
    """

    # Transformed 3D coordinates (N, 3) arrays
    pixel_points: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None

    # Region labels for each point/centroid
    points_labels: Optional[np.ndarray] = None
    centroids_labels: Optional[np.ndarray] = None

    # Hemisphere labels for each point/centroid
    points_hemi_labels: Optional[np.ndarray] = None
    centroids_hemi_labels: Optional[np.ndarray] = None

    # Per-section counts (number of points/centroids per section)
    points_len: Optional[List[int]] = None
    centroids_len: Optional[List[int]] = None

    # Source filenames for each section
    segmentation_filenames: Optional[List[str]] = None

    # Region areas per section (for density calculations)
    region_areas_list: Optional[List[pd.DataFrame]] = None

    # Damage mask flags per point/centroid
    per_point_undamaged: Optional[np.ndarray] = None
    per_centroid_undamaged: Optional[np.ndarray] = None

    # Custom region labels (if custom_region_path was specified)
    points_custom_labels: Optional[np.ndarray] = None
    centroids_custom_labels: Optional[np.ndarray] = None


@dataclass
class IntensityCoordinateResults:
    """Results from intensity-mode coordinate extraction.

    Similar to CoordinateResults but includes intensity values instead of centroids.
    """

    # Transformed 3D coordinates (N, 3) array
    pixel_points: Optional[np.ndarray] = None

    # Region labels and hemisphere labels
    points_labels: Optional[np.ndarray] = None
    points_hemi_labels: Optional[np.ndarray] = None

    # Per-section counts
    points_len: Optional[List[int]] = None

    # Source filenames
    segmentation_filenames: Optional[List[str]] = None

    # Intensity values per point
    point_intensities: Optional[np.ndarray] = None

    # Region-level intensity aggregations per section
    region_intensities_list: Optional[List[pd.DataFrame]] = None


@dataclass
class QuantificationResults:
    """Results from quantification (quantify_coordinates).

    Contains aggregated counts and statistics by atlas region.
    """

    # Whole-series aggregated DataFrame with columns:
    # idx, name, r, g, b, pixel_count, centroid_count, area_um2, etc.
    label_df: Optional[pd.DataFrame] = None

    # List of per-section DataFrames with the same structure
    per_section_df: Optional[List[pd.DataFrame]] = None

    # Custom region results (if custom_region_path was specified)
    custom_label_df: Optional[pd.DataFrame] = None
    custom_per_section_df: Optional[List[pd.DataFrame]] = None


@dataclass
class VolumeResults:
    """Results from volume interpolation (interpolate_volume).

    Contains 3D volumes projected from section data.
    """

    # Interpolated signal volume (float32)
    # Values represent mean/count depending on value_mode parameter
    interpolated_volume: Optional[np.ndarray] = None

    # Frequency volume (uint32)
    # Number of section pixels contributing to each voxel
    frequency_volume: Optional[np.ndarray] = None

    # Damage volume (float32)
    # Accumulated damage mask projected to 3D
    damage_volume: Optional[np.ndarray] = None


@dataclass
class AnalysisState:
    """Complete state of a PyNutil analysis.

    This dataclass aggregates all results from the analysis pipeline,
    providing a single container for the entire analysis state.
    """

    # Configuration
    segmentation_folder: Optional[str] = None
    image_folder: Optional[str] = None
    alignment_json: Optional[str] = None
    colour: Optional[List[int]] = None
    intensity_channel: Optional[str] = None
    atlas_name: Optional[str] = None
    voxel_size_um: Optional[float] = None

    # Atlas data
    atlas_volume: Optional[np.ndarray] = None
    hemi_map: Optional[np.ndarray] = None
    atlas_labels: Optional[pd.DataFrame] = None

    # Custom regions
    custom_regions_dict: Optional[Dict[str, Any]] = None
    custom_atlas_labels: Optional[pd.DataFrame] = None

    # Pipeline results
    coordinate_results: Optional[CoordinateResults] = None
    intensity_results: Optional[IntensityCoordinateResults] = None
    quantification_results: Optional[QuantificationResults] = None
    volume_results: Optional[VolumeResults] = None

    # Processing flags
    apply_damage_mask: bool = True

    @property
    def is_intensity_mode(self) -> bool:
        """Check if running in intensity measurement mode."""
        return self.image_folder is not None and self.segmentation_folder is None

    @property
    def has_coordinates(self) -> bool:
        """Check if coordinates have been extracted."""
        return (
            self.coordinate_results is not None
            or self.intensity_results is not None
        )

    @property
    def has_quantification(self) -> bool:
        """Check if quantification has been performed."""
        return self.quantification_results is not None

    @property
    def has_volumes(self) -> bool:
        """Check if volume interpolation has been performed."""
        return self.volume_results is not None

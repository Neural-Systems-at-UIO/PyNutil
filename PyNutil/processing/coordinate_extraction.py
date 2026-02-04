"""Coordinate extraction for mapping segmentations to atlas space.

This module provides the public API for extracting pixel and centroid
coordinates from segmentation images and transforming them to atlas space.

The implementation is split across:
- connected_components: Low-level connected component analysis
- section_processor: Single section processing
- batch_processor: Folder-level batch processing
"""

# Re-export public API from submodules
from .batch_processor import (
    folder_to_atlas_space,
    folder_to_atlas_space_intensity,
    create_threads,
)
from .section_processor import (
    segmentation_to_atlas_space,
    segmentation_to_atlas_space_intensity,
    get_centroids,
    get_scaled_pixels,
)
from .connected_components import (
    connected_components_props,
    labeled_image_props,
    get_centroids_and_area,
    get_objects_and_assign_regions,
)

__all__ = [
    # batch_processor
    "folder_to_atlas_space",
    "folder_to_atlas_space_intensity",
    "create_threads",
    # section_processor
    "segmentation_to_atlas_space",
    "segmentation_to_atlas_space_intensity",
    "get_centroids",
    "get_scaled_pixels",
    # connected_components
    "connected_components_props",
    "labeled_image_props",
    "get_centroids_and_area",
    "get_objects_and_assign_regions",
]

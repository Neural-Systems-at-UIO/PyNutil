from __future__ import annotations

from typing import Any, Dict

from . import counting_and_load
from .visualign_deformations import triangulate


def get_triangulation(
    slice_dict: Dict[str, Any],
    reg_width: int,
    reg_height: int,
    non_linear: bool,
):
    """Generate triangulation data if non-linear markers exist."""

    if non_linear and "markers" in slice_dict:
        return triangulate(reg_width, reg_height, slice_dict["markers"])
    return None


def get_region_areas(
    use_flat,
    atlas_labels,
    flat_file_atlas,
    seg_width,
    seg_height,
    slice_dict,
    atlas_volume,
    hemi_mask,
    triangulation,
    damage_mask,
):
    """Build the atlas map for a slice and compute region areas.

    This performs the atlas slice extraction (from volume or flat file), applies
    non-linear warping if requested (via `triangulation`), and converts the
    resulting label map into a region-area dataframe.
    """

    reg_width, reg_height = slice_dict["width"], slice_dict["height"]
    atlas_map = counting_and_load.load_image(
        flat_file_atlas,
        slice_dict["anchoring"],
        atlas_volume,
        triangulation,
        (reg_width, reg_height),
        atlas_labels,
    )
    region_areas = counting_and_load.flat_to_dataframe(
        atlas_map, damage_mask, hemi_mask, (seg_width, seg_height)
    )
    return region_areas, atlas_map

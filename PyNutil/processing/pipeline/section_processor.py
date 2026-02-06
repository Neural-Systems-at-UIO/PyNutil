"""Single section processing for atlas space transformation.

This module contains functions for transforming individual segmentation
files or images into atlas space coordinates.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import cv2

from ...io.loaders import load_segmentation
from ...results import SectionResult, IntensitySectionResult
from .connected_components import (
    get_objects_and_assign_regions,
)
from ..atlas_map import generate_target_slice, get_region_areas
from ..transforms import (
    transform_points_to_atlas_space,
    transform_to_registration,
    get_transformed_coordinates,
    transform_to_atlas_space,
)
from ..adapters.segmentation import detect_pixel_id
from ..analysis.aggregator import build_region_intensity_dataframe
from ..utils import (
    convert_to_intensity,
    assign_labels_at_coordinates,
    resize_mask_nearest,
)

if TYPE_CHECKING:
    from ..adapters import SliceInfo


def _prepare_section(
    slice_info,
    seg_width,
    seg_height,
    atlas_labels,
    flat_file_atlas,
    atlas_volume,
    hemi_map,
    non_linear,
    use_flat,
):
    """Shared setup for both segmentation and intensity section processors.

    Returns:
        (atlas_map, region_areas, hemi_mask, damage_mask, deformation)
    """
    deformation = slice_info.deformation if non_linear else None
    damage_mask = slice_info.damage_mask
    reg_height, reg_width = slice_info.height, slice_info.width

    if hemi_map is not None:
        hemi_mask = generate_target_slice(slice_info.anchoring, hemi_map)
    else:
        hemi_mask = None

    region_areas, atlas_map = get_region_areas(
        use_flat,
        atlas_labels,
        flat_file_atlas,
        seg_width,
        seg_height,
        slice_info.anchoring,
        reg_width,
        reg_height,
        atlas_volume,
        hemi_mask,
        deformation,
        damage_mask,
    )
    return atlas_map, region_areas, hemi_mask, damage_mask, deformation


# ---------------------------------------------------------------------------
# Helpers for segmentation_to_atlas_space
# ---------------------------------------------------------------------------


def _compute_damage_state(
    damage_mask,
    scaled_x,
    scaled_y,
    scaled_centroidsX,
    scaled_centroidsY,
    reg_width,
    reg_height,
):
    """Compute per-point and per-centroid undamaged boolean masks."""
    if damage_mask is not None:
        damage_mask = resize_mask_nearest(damage_mask, reg_width, reg_height).astype(
            bool
        )
        per_point_undamaged = damage_mask[
            np.round(scaled_y).astype(int), np.round(scaled_x).astype(int)
        ]
        if scaled_centroidsX is not None and scaled_centroidsY is not None:
            per_centroid_undamaged = damage_mask[
                np.round(scaled_centroidsY).astype(int),
                np.round(scaled_centroidsX).astype(int),
            ]
        else:
            per_centroid_undamaged = np.array([], dtype=bool)
    else:
        per_point_undamaged = np.ones(scaled_x.shape, dtype=bool)
        if scaled_centroidsX is not None:
            per_centroid_undamaged = np.ones(scaled_centroidsX.shape, dtype=bool)
        else:
            per_centroid_undamaged = np.array([], dtype=bool)
    return per_point_undamaged, per_centroid_undamaged


def _compute_hemi_state(
    hemi_mask,
    scaled_x,
    scaled_y,
    scaled_centroidsX,
    scaled_centroidsY,
    reg_height,
    reg_width,
):
    """Compute per-point and per-centroid hemisphere labels."""
    if hemi_mask is not None:
        per_point_hemi = assign_labels_at_coordinates(
            scaled_y, scaled_x, hemi_mask, reg_height, reg_width
        )
        if scaled_centroidsX is not None and scaled_centroidsY is not None:
            per_centroid_hemi = assign_labels_at_coordinates(
                scaled_centroidsY, scaled_centroidsX, hemi_mask, reg_height, reg_width
            )
        else:
            per_centroid_hemi = np.array([], dtype=hemi_mask.dtype)
    else:
        per_point_hemi = [None] * len(scaled_x)
        per_centroid_hemi = [None] * (
            len(scaled_centroidsX) if scaled_centroidsX is not None else 0
        )
    return per_point_hemi, per_centroid_hemi


def _safe_index(arr, mask):
    """Index *arr* by boolean *mask*, returning empty array when *arr* is None."""
    if arr is not None:
        return arr[mask]
    return np.array([])


def _to_array(val, gate):
    """Return ``np.array(val)`` when *gate* is not None, else empty array."""
    if gate is not None:
        return np.array(val)
    return np.array([])


# ---------------------------------------------------------------------------
# Helpers for segmentation_to_atlas_space_intensity
# ---------------------------------------------------------------------------


def _apply_intensity_bounds(arr, min_val, max_val):
    """Zero out pixels outside ``[min_val, max_val]`` bounds, in-place."""
    if min_val is not None:
        arr[arr < min_val] = 0
    if max_val is not None:
        arr[arr > max_val] = 0


def _filter_rgb_by_intensity(image_resized, min_intensity, max_intensity):
    """Zero RGB pixels whose luminance falls outside intensity bounds."""
    if min_intensity is None and max_intensity is None:
        return
    temp_gray = (
        0.2989 * image_resized[:, :, 0]
        + 0.5870 * image_resized[:, :, 1]
        + 0.1140 * image_resized[:, :, 2]
    )
    if min_intensity is not None:
        image_resized[temp_gray < min_intensity] = 0
    if max_intensity is not None:
        image_resized[temp_gray > max_intensity] = 0


def _extract_signal_pixels(
    image,
    intensity_resized,
    signal_mask,
    reg_width,
    reg_height,
    min_intensity,
    max_intensity,
):
    """Extract and subsample signal pixels for MeshView."""
    sig_y, sig_x = np.where(signal_mask)

    if image.ndim == 3:
        image_resized = cv2.resize(
            image, (reg_width, reg_height), interpolation=cv2.INTER_AREA
        )
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        _filter_rgb_by_intensity(image_resized, min_intensity, max_intensity)
        sig_intensities = image_resized[sig_y, sig_x]
    else:
        sig_intensities = intensity_resized[sig_y, sig_x]

    # Sample pixels to keep MeshView responsive
    max_points = 100000
    if len(sig_y) > max_points:
        indices = np.linspace(0, len(sig_y) - 1, max_points).astype(int)
        sig_y = sig_y[indices]
        sig_x = sig_x[indices]
        sig_intensities = sig_intensities[indices]

    return sig_y, sig_x, sig_intensities


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segmentation_to_atlas_space(
    slice_info: "SliceInfo",
    segmentation_path,
    atlas_labels,
    flat_file_atlas=None,
    pixel_id="auto",
    non_linear=True,
    object_cutoff=0,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    segmentation_format="binary",
):
    """Transform a single segmentation file into atlas space.

    Args:
        slice_info: SliceInfo object from registration adapter.
        segmentation_path: Path to the segmentation file.
        atlas_labels: Atlas labels DataFrame.
        flat_file_atlas: Path to flat atlas (optional).
        pixel_id: Pixel color or 'auto' for auto-detection.
        non_linear: Use non-linear transforms if True.
        object_cutoff: Minimum object size.
        atlas_volume: 3D atlas volume.
        hemi_map: Hemisphere mask.
        use_flat: Use flat files if True.
        segmentation_format: Format name ("binary" or "cellpose").

    Returns:
        SectionResult with transformed coordinates, labels, and metadata.
    """
    segmentation = load_segmentation(segmentation_path)
    if pixel_id == "auto" and segmentation_format == "binary":
        pixel_id = detect_pixel_id(segmentation)
    seg_height, seg_width = segmentation.shape[:2]
    reg_height, reg_width = slice_info.height, slice_info.width

    atlas_map, region_areas, hemi_mask, damage_mask, deformation = _prepare_section(
        slice_info,
        seg_width,
        seg_height,
        atlas_labels,
        flat_file_atlas,
        atlas_volume,
        hemi_map,
        non_linear,
        use_flat,
    )
    y_scale, x_scale = transform_to_registration(
        seg_height, seg_width, reg_height, reg_width
    )

    (
        centroids,
        scaled_centroidsX,
        scaled_centroidsY,
        scaled_y,
        scaled_x,
        per_centroid_labels,
    ) = get_objects_and_assign_regions(
        segmentation,
        pixel_id,
        atlas_map,
        y_scale,
        x_scale,
        object_cutoff=object_cutoff,
        atlas_at_original_resolution=True,
        reg_height=reg_height,
        reg_width=reg_width,
        segmentation_format=segmentation_format,
    )

    if scaled_y is None or scaled_x is None:
        gc.collect()
        return SectionResult.empty(region_areas)

    per_point_labels = assign_labels_at_coordinates(
        scaled_y, scaled_x, atlas_map, reg_height, reg_width
    )
    per_point_undamaged, per_centroid_undamaged = _compute_damage_state(
        damage_mask,
        scaled_x,
        scaled_y,
        scaled_centroidsX,
        scaled_centroidsY,
        reg_width,
        reg_height,
    )
    per_point_hemi, per_centroid_hemi = _compute_hemi_state(
        hemi_mask,
        scaled_x,
        scaled_y,
        scaled_centroidsX,
        scaled_centroidsY,
        reg_height,
        reg_width,
    )

    if per_centroid_labels is None:
        per_centroid_labels = np.array([], dtype=per_point_labels.dtype)

    new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(
        non_linear,
        None,
        scaled_x[per_point_undamaged],
        scaled_y[per_point_undamaged],
        _safe_index(scaled_centroidsX, per_centroid_undamaged),
        _safe_index(scaled_centroidsY, per_centroid_undamaged),
        deformation,
    )
    points, centroids = transform_points_to_atlas_space(
        slice_info.anchoring,
        new_x,
        new_y,
        centroids_new_x,
        centroids_new_y,
        reg_height,
        reg_width,
    )
    result = SectionResult(
        points=_to_array(points, points),
        centroids=_to_array(centroids, centroids),
        region_areas=region_areas,
        points_labels=_to_array(per_point_labels, points),
        centroids_labels=_to_array(per_centroid_labels, centroids),
        per_point_undamaged=_to_array(per_point_undamaged, points),
        per_centroid_undamaged=_to_array(per_centroid_undamaged, centroids),
        points_hemi_labels=_to_array(per_point_hemi, points),
        centroids_hemi_labels=_to_array(per_centroid_hemi, points),
    )

    gc.collect()
    return result


def segmentation_to_atlas_space_intensity(
    slice_info: "SliceInfo",
    image_path,
    atlas_labels,
    intensity_channel,
    flat_file_atlas=None,
    non_linear=True,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    min_intensity=None,
    max_intensity=None,
):
    """Transform a single image file into atlas space and extract intensity.

    Args:
        slice_info: SliceInfo object from registration adapter.
        image_path: Path to the image file.
        atlas_labels: Atlas labels DataFrame.
        intensity_channel: Channel to use for intensity.
        flat_file_atlas: Path to flat atlas (optional).
        non_linear: Apply non-linear transform.
        atlas_volume: 3D atlas volume.
        hemi_map: Hemisphere mask.
        use_flat: Use flat files if True.
        min_intensity: Minimum intensity threshold.
        max_intensity: Maximum intensity threshold.

    Returns:
        IntensitySectionResult with region intensities and MeshView point data.
    """
    image = load_segmentation(image_path)
    intensity = convert_to_intensity(image, intensity_channel)
    _apply_intensity_bounds(intensity, min_intensity, max_intensity)

    reg_height, reg_width = slice_info.height, slice_info.width

    atlas_map, region_areas, hemi_mask, damage_mask, deformation = _prepare_section(
        slice_info,
        image.shape[1],
        image.shape[0],
        atlas_labels,
        flat_file_atlas,
        atlas_volume,
        hemi_map,
        non_linear,
        use_flat,
    )

    # Ensure atlas_map and hemi_mask match registration resolution
    atlas_map = atlas_map.astype(np.int64)
    if atlas_map.shape != (reg_height, reg_width):
        atlas_map = cv2.resize(
            atlas_map.astype(np.float32),
            (reg_width, reg_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    if hemi_mask is not None and hemi_mask.shape != (reg_height, reg_width):
        hemi_mask = resize_mask_nearest(hemi_mask, reg_width, reg_height)

    # Resize intensity to registration space to match atlas_map
    intensity_resized = cv2.resize(
        intensity, (reg_width, reg_height), interpolation=cv2.INTER_AREA
    )
    _apply_intensity_bounds(intensity_resized, min_intensity, max_intensity)

    # Apply damage mask if it exists
    if damage_mask is not None:
        damage_mask_resized = resize_mask_nearest(
            damage_mask, reg_width, reg_height
        ).astype(bool)
        intensity_resized[~damage_mask_resized] = 0
        atlas_map[~damage_mask_resized] = 0
    else:
        damage_mask_resized = None

    df = build_region_intensity_dataframe(
        atlas_map=atlas_map,
        intensity_resized=intensity_resized,
        atlas_labels=atlas_labels,
        region_areas=region_areas,
        hemi_mask=hemi_mask,
        damage_mask_resized=damage_mask_resized,
    )

    # Build signal mask and extract pixels for MeshView
    signal_mask = np.ones_like(intensity_resized, dtype=bool)
    if min_intensity is not None:
        signal_mask &= intensity_resized >= min_intensity
    if max_intensity is not None:
        signal_mask &= intensity_resized <= max_intensity
    if damage_mask is not None:
        signal_mask &= damage_mask_resized

    sig_y, sig_x, sig_intensities = _extract_signal_pixels(
        image,
        intensity_resized,
        signal_mask,
        reg_width,
        reg_height,
        min_intensity,
        max_intensity,
    )

    result = _build_intensity_result(
        df,
        sig_y,
        sig_x,
        sig_intensities,
        slice_info,
        atlas_map,
        hemi_mask,
        reg_height,
        reg_width,
    )

    gc.collect()
    return result


def _build_intensity_result(
    df,
    sig_y,
    sig_x,
    sig_intensities,
    slice_info,
    atlas_map,
    hemi_mask,
    reg_height,
    reg_width,
):
    """Construct an IntensitySectionResult from extracted signal pixels."""
    if len(sig_y) == 0:
        return IntensitySectionResult(
            region_intensities=df,
            points=None,
            points_labels=None,
            points_hemi_labels=None,
            point_intensities=None,
            num_points=0,
        )

    sig_points_3d = transform_to_atlas_space(
        slice_info.anchoring,
        sig_y,
        sig_x,
        reg_height,
        reg_width,
    )
    sig_labels = atlas_map[sig_y, sig_x]
    sig_hemi = (
        hemi_mask[sig_y, sig_x]
        if hemi_mask is not None
        else np.zeros(len(sig_y), dtype=int)
    )

    return IntensitySectionResult(
        region_intensities=df,
        points=sig_points_3d,
        points_labels=sig_labels,
        points_hemi_labels=sig_hemi,
        point_intensities=sig_intensities,
        num_points=len(sig_y),
    )

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
from ...results import SectionResult
from .connected_components import (
    get_centroids_and_area,
    get_objects_and_assign_regions,
)
from ..generate_target_slice import generate_target_slice
from ..transforms import (
    transform_points_to_atlas_space,
    transform_to_registration,
    get_transformed_coordinates,
    transform_to_atlas_space,
    get_region_areas,
)
from .image_loaders import detect_pixel_id
from ..analysis.aggregator import build_region_intensity_dataframe
from ..utils import (
    find_matching_pixels,
    scale_positions,
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
        slice_info, seg_width, seg_height, atlas_labels, flat_file_atlas,
        atlas_volume, hemi_map, non_linear, use_flat,
    )
    y_scale, x_scale = transform_to_registration(
        seg_height, seg_width, reg_height, reg_width
    )

    centroids, points = None, None
    scaled_centroidsX, scaled_centroidsY, scaled_x, scaled_y = None, None, None, None

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

    # Assign per-pixel labels using atlas at original resolution (scale coords down)
    per_point_labels = assign_labels_at_coordinates(
        scaled_y, scaled_x, atlas_map, reg_height, reg_width
    )
    if damage_mask is not None:
        damage_mask = resize_mask_nearest(damage_mask, reg_width, reg_height).astype(bool)
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
    if hemi_mask is not None:
        # Assign hemisphere labels in atlas space (scale coords down)
        per_point_hemi = assign_labels_at_coordinates(
            scaled_y, scaled_x, hemi_mask, reg_height, reg_width
        )

        if scaled_centroidsX is not None and scaled_centroidsY is not None:
            per_centroid_hemi = assign_labels_at_coordinates(
                scaled_centroidsY, scaled_centroidsX, hemi_mask, reg_height, reg_width
            )
        else:
            per_centroid_hemi = np.array([], dtype=hemi_mask.dtype)
        per_point_hemi = per_point_hemi[per_point_undamaged]
        per_centroid_hemi = per_centroid_hemi[per_centroid_undamaged]
    else:
        per_point_hemi = [None] * len(scaled_x)
        per_centroid_hemi = [None] * (
            len(scaled_centroidsX) if scaled_centroidsX is not None else 0
        )

    per_point_labels = per_point_labels[per_point_undamaged]
    if per_centroid_labels is None:
        per_centroid_labels = np.array([], dtype=per_point_labels.dtype)
    per_centroid_labels = per_centroid_labels[per_centroid_undamaged]

    new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(
        non_linear,
        None,  # slice_dict no longer needed
        scaled_x[per_point_undamaged],
        scaled_y[per_point_undamaged],
        (
            scaled_centroidsX[per_centroid_undamaged]
            if scaled_centroidsX is not None
            else np.array([])
        ),
        (
            scaled_centroidsY[per_centroid_undamaged]
            if scaled_centroidsY is not None
            else np.array([])
        ),
        deformation,  # Pass deformation function instead of triangulation
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
        points=np.array(points if points is not None else []),
        centroids=np.array(centroids if centroids is not None else []),
        region_areas=region_areas,
        points_labels=np.array(per_point_labels if points is not None else []),
        centroids_labels=np.array(
            per_centroid_labels if centroids is not None else []
        ),
        per_point_undamaged=np.array(
            per_point_undamaged if points is not None else []
        ),
        per_centroid_undamaged=np.array(
            per_centroid_undamaged if centroids is not None else []
        ),
        points_hemi_labels=np.array(per_point_hemi if points is not None else []),
        centroids_hemi_labels=np.array(
            per_centroid_hemi if points is not None else []
        ),
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
    region_intensities_list=None,
    index=None,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    centroids_list=None,
    centroids_labels_list=None,
    centroids_hemi_labels_list=None,
    centroids_intensities_list=None,
    centroids_len=None,
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
        region_intensities_list: Storage for intensity data.
        index: Index in output lists.
        atlas_volume: 3D atlas volume.
        hemi_map: Hemisphere mask.
        use_flat: Use flat files if True.
        centroids_list: Storage for pixel coordinates.
        centroids_labels_list: Storage for pixel labels.
        centroids_hemi_labels_list: Storage for hemisphere labels.
        centroids_intensities_list: Storage for intensity values.
        centroids_len: Storage for point counts.
        min_intensity: Minimum intensity threshold.
        max_intensity: Maximum intensity threshold.

    Returns:
        None (results stored in provided lists)
    """
    image = load_segmentation(image_path)
    intensity = convert_to_intensity(image, intensity_channel)

    # Apply intensity filters if specified
    if min_intensity is not None:
        intensity[intensity < min_intensity] = 0
    if max_intensity is not None:
        intensity[intensity > max_intensity] = 0

    reg_height, reg_width = slice_info.height, slice_info.width

    atlas_map, region_areas, hemi_mask, damage_mask, deformation = _prepare_section(
        slice_info, image.shape[1], image.shape[0], atlas_labels, flat_file_atlas,
        atlas_volume, hemi_map, non_linear, use_flat,
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

    # Re-apply intensity filters after resizing to handle interpolation artifacts
    if min_intensity is not None:
        intensity_resized[intensity_resized < min_intensity] = 0
    if max_intensity is not None:
        intensity_resized[intensity_resized > max_intensity] = 0

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

    region_intensities_list[index] = df

    # Extract pixels for MeshView
    # We respect the intensity filters if specified
    signal_mask = np.ones_like(intensity_resized, dtype=bool)
    if min_intensity is not None:
        signal_mask &= (intensity_resized >= min_intensity)
    if max_intensity is not None:
        signal_mask &= (intensity_resized <= max_intensity)

    if damage_mask is not None:
        signal_mask &= damage_mask_resized

    sig_y, sig_x = np.where(signal_mask)

    # If image is RGB, we want to keep the original colors for MeshView
    if image.ndim == 3:
        image_resized = cv2.resize(image, (reg_width, reg_height), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB for MeshView
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Apply intensity filters to RGB data as well to ensure consistency
        if min_intensity is not None or max_intensity is not None:
            # Calculate grayscale for filtering
            temp_gray = (0.2989 * image_resized[:, :, 0] + 0.5870 * image_resized[:, :, 1] + 0.1140 * image_resized[:, :, 2])
            if min_intensity is not None:
                image_resized[temp_gray < min_intensity] = 0
            if max_intensity is not None:
                image_resized[temp_gray > max_intensity] = 0

        sig_intensities = image_resized[sig_y, sig_x]
    else:
        sig_intensities = intensity_resized[sig_y, sig_x]

    # Sample pixels to keep MeshView responsive
    # 100,000 points per section is a good balance
    max_points = 100000
    if len(sig_y) > max_points:
        # Use linspace for reproducible sampling
        indices = np.linspace(0, len(sig_y) - 1, max_points).astype(int)
        sig_y = sig_y[indices]
        sig_x = sig_x[indices]
        sig_intensities = sig_intensities[indices]

    if len(sig_y) > 0:
        # Transform to 3D atlas space
        sig_points_3d = transform_to_atlas_space(
            slice_info.anchoring,
            sig_y,
            sig_x,
            reg_height,
            reg_width,
        )

        # Get atlas labels and hemi labels for these points
        sig_labels = atlas_map[sig_y, sig_x]
        if hemi_mask is not None:
            sig_hemi = hemi_mask[sig_y, sig_x]
        else:
            sig_hemi = np.zeros(len(sig_y), dtype=int)

        # Store results
        if centroids_list is not None:
            centroids_list[index] = sig_points_3d
            centroids_labels_list[index] = sig_labels
            centroids_hemi_labels_list[index] = sig_hemi
            centroids_intensities_list[index] = sig_intensities
            centroids_len[index] = len(sig_y)
    else:
        if centroids_len is not None:
            centroids_len[index] = 0
    gc.collect()


def get_centroids(segmentation, pixel_id, y_scale, x_scale, object_cutoff=0):
    """Find object centroids for a given pixel color and apply scaling.

    Args:
        segmentation: Segmentation array.
        pixel_id: Pixel color to match.
        y_scale: Vertical scaling factor.
        x_scale: Horizontal scaling factor.
        object_cutoff: Minimum object size.

    Returns:
        tuple: (centroids, scaled_centroidsX, scaled_centroidsY)
    """
    binary_seg = segmentation == pixel_id
    binary_seg = np.all(binary_seg, axis=2)
    centroids, area, coords = get_centroids_and_area(
        binary_seg, pixel_cut_off=object_cutoff
    )
    if len(centroids) == 0:
        return None, None, None
    centroidsX = centroids[:, 1]
    centroidsY = centroids[:, 0]
    scaled_centroidsY, scaled_centroidsX = scale_positions(
        centroidsY, centroidsX, y_scale, x_scale
    )
    return centroids, scaled_centroidsX, scaled_centroidsY


def get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale):
    """Retrieve pixel coordinates for a specified color and scale them.

    Args:
        segmentation: Segmentation array.
        pixel_id: Pixel color to match.
        y_scale: Vertical scaling factor.
        x_scale: Horizontal scaling factor.

    Returns:
        tuple: (scaled_y, scaled_x)
    """
    id_pixels = find_matching_pixels(segmentation, pixel_id)
    if len(id_pixels[0]) == 0:
        return None, None
    scaled_y, scaled_x = scale_positions(id_pixels[0], id_pixels[1], y_scale, x_scale)
    return scaled_y, scaled_x

"""Connected component analysis for segmentation images.

This module provides functions for detecting and analyzing connected
components in binary segmentation masks, including centroid extraction
and region assignment.
"""

import numpy as np
import cv2

from ..utils import scale_positions, assign_labels_at_coordinates


# ── Shared pixel-grouping helper ─────────────────────────────────────────


def _group_pixels_by_label(ys, xs, labels, *, compute_centroid=True):
    """Group pixel coordinates by their label ID.

    Shared implementation used by both :func:`connected_components_props`
    (OpenCV labels) and :func:`labeled_image_props` (pre-labelled images).

    Args:
        ys: 1-D array of Y coordinates.
        xs: 1-D array of X coordinates.
        labels: 1-D array of integer label IDs (same length as *ys*).
        compute_centroid: If ``True`` compute the mean centroid; if ``False``
            the ``centroid`` key in each dict will be ``None``.

    Returns:
        list[dict]: One dict per unique label with keys ``area``, ``centroid``
        (y, x) and ``coords`` (N × 2 ndarray).
    """
    if ys.size == 0:
        return []

    order = np.argsort(labels, kind="stable")
    ys = ys[order]
    xs = xs[order]
    labels = labels[order]

    unique_ids, start_idx = np.unique(labels, return_index=True)
    end_idx = np.r_[start_idx[1:], labels.size]

    props = []
    for _, start, end in zip(unique_ids, start_idx, end_idx):
        comp_ys = ys[start:end]
        comp_xs = xs[start:end]
        area = int(end - start)
        centroid = (
            (float(np.mean(comp_ys)), float(np.mean(comp_xs)))
            if compute_centroid
            else None
        )
        coords = np.column_stack((comp_ys, comp_xs))
        props.append({"area": area, "centroid": centroid, "coords": coords})
    return props


def connected_components_props(binary_mask: np.ndarray, *, connectivity: int = 4):
    """Return connected-component properties for a boolean 2D mask.

    Uses OpenCV so we can avoid scikit-image dependency.

    Args:
        binary_mask: Boolean 2D array representing the mask.
        connectivity: Pixel connectivity (4 or 8).

    Returns:
        list: List of dicts with:
            - area (int)
            - centroid (tuple[float, float]) in (y, x) order
            - coords (ndarray[int]) of shape (N, 2) in (y, x) order
    """
    if binary_mask.size == 0:
        return []

    binary_u8 = binary_mask.astype(np.uint8, copy=False)
    num_labels, labels, stats, centroids_xy = cv2.connectedComponentsWithStats(
        binary_u8, connectivity=connectivity
    )
    if num_labels <= 1:
        return []

    ys, xs = np.nonzero(labels)
    if ys.size == 0:
        return []
    comp_ids = labels[ys, xs]

    # Use shared grouping; override centroids with OpenCV's sub-pixel values
    props = _group_pixels_by_label(ys, xs, comp_ids, compute_centroid=False)

    # Map grouped label IDs back to OpenCV stats/centroids
    order = np.argsort(comp_ids, kind="stable")
    unique_ids = np.unique(comp_ids[order])
    for prop, comp_id in zip(props, unique_ids):
        cx, cy = centroids_xy[comp_id]
        prop["centroid"] = (float(cy), float(cx))
        prop["area"] = int(stats[comp_id, cv2.CC_STAT_AREA])
    return props


def labeled_image_props(label_image: np.ndarray):
    """Extract properties from a pre-labeled image (e.g., Cellpose output).

    Args:
        label_image: 2D array where each unique non-zero value is an object.

    Returns:
        list: List of dicts with area, centroid, and coords for each object.
    """
    ys, xs = np.nonzero(label_image)
    if ys.size == 0:
        return []
    labels = label_image[ys, xs]
    return _group_pixels_by_label(ys, xs, labels)


def get_objects_and_assign_regions(
    segmentation,
    pixel_id,
    atlas_map,
    y_scale,
    x_scale,
    object_cutoff=0,
    atlas_at_original_resolution=False,
    reg_height=None,
    reg_width=None,
    segmentation_format="binary",
):
    """Single-pass object detection, pixel extraction, and region assignment.

    Args:
        segmentation: Segmentation image array.
        pixel_id: Pixel color to match [R, G, B] (only used for binary format).
        atlas_map: 2D atlas label map.
        y_scale: Vertical scaling factor.
        x_scale: Horizontal scaling factor.
        object_cutoff: Minimum object size.
        atlas_at_original_resolution: If True, scale coords to atlas resolution.
        reg_height: Registration height (required if atlas_at_original_resolution).
        reg_width: Registration width (required if atlas_at_original_resolution).
        segmentation_format: Format name ("binary" or "cellpose").

    Returns:
        tuple: (centroids, scaled_centroidsX, scaled_centroidsY,
                scaled_y, scaled_x, per_centroid_labels)
    """
    from ..adapters import SegmentationAdapterRegistry

    adapter = SegmentationAdapterRegistry.get(segmentation_format)

    # Create binary mask using the adapter
    binary_seg = adapter.create_binary_mask(segmentation, pixel_id)

    # Get pixel coordinates
    pixel_y, pixel_x = np.where(binary_seg)

    if len(pixel_y) == 0:
        del binary_seg
        return None, None, None, None, None, None

    # Scale pixel coordinates to registration space
    scaled_y, scaled_x = scale_positions(pixel_y, pixel_x, y_scale, x_scale)

    # Extract objects using the adapter
    objects_info = adapter.extract_objects(
        segmentation, binary_seg, min_area=object_cutoff
    )

    if len(objects_info) == 0:
        return None, None, None, scaled_y, scaled_x, None

    centroids = []
    per_centroid_labels = []

    for obj in objects_info:
        centroids.append(obj.centroid)
        # Scale object coords
        scaled_obj_y, scaled_obj_x = scale_positions(
            obj.coords[:, 0], obj.coords[:, 1], y_scale, x_scale
        )

        # Handle resolution scaling strategy
        if atlas_at_original_resolution:
            atlas_height, atlas_width = atlas_map.shape
            assignment_y = scaled_obj_y * (atlas_height / reg_height)
            assignment_x = scaled_obj_x * (atlas_width / reg_width)
        else:
            assignment_y, assignment_x = scaled_obj_y, scaled_obj_x

        # Check bounds
        iy = np.round(assignment_y).astype(int)
        ix = np.round(assignment_x).astype(int)
        valid_mask = (
            (iy >= 0)
            & (iy < atlas_map.shape[0])
            & (ix >= 0)
            & (ix < atlas_map.shape[1])
        )

        if not np.any(valid_mask):
            per_centroid_labels.append(0)
            continue

        # Assign region based on majority vote of pixels in object
        pixel_labels = atlas_map[iy[valid_mask], ix[valid_mask]]
        unique_labels, counts = np.unique(pixel_labels, return_counts=True)
        per_centroid_labels.append(unique_labels[np.argmax(counts)])

    centroids = np.array(centroids)
    scaled_centroidsY, scaled_centroidsX = scale_positions(
        centroids[:, 0], centroids[:, 1], y_scale, x_scale
    )
    per_centroid_labels = np.array(per_centroid_labels)

    return (
        centroids,
        scaled_centroidsX,
        scaled_centroidsY,
        scaled_y,
        scaled_x,
        per_centroid_labels,
    )

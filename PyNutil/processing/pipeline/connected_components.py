"""Connected component analysis for segmentation images.

This module provides functions for detecting and analyzing connected
components in binary segmentation masks, including centroid extraction
and region assignment.
"""

import numpy as np
import cv2


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
        tuple: (props, unique_ids) where *props* is a list[dict] with keys
        ``area``, ``centroid`` (y, x) and ``coords`` (N × 2 ndarray), and
        *unique_ids* is the sorted array of label IDs.
    """
    if ys.size == 0:
        return [], np.array([], dtype=labels.dtype)

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
    return props, unique_ids


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
    props, unique_ids = _group_pixels_by_label(ys, xs, comp_ids, compute_centroid=False)

    # Map grouped label IDs back to OpenCV stats/centroids
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
    props, _ = _group_pixels_by_label(ys, xs, labels)
    return props


def get_objects_and_assign_regions(
    segmentation,
    adapter,
    pixel_id,
    atlas_map,
    y_scale,
    x_scale,
    object_cutoff=0,
):
    """Single-pass object detection, pixel extraction, and region assignment.

    Args:
        segmentation: Segmentation image array.
        adapter: Segmentation adapter instance.
        pixel_id: Pixel color to match [R, G, B] (only used for binary format).
        atlas_map: 2D atlas label map.
        y_scale: Vertical scaling factor (segmentation → registration space).
        x_scale: Horizontal scaling factor (segmentation → registration space).
        object_cutoff: Minimum object size.

    Returns:
        tuple: (centroids, scaled_centroidsX, scaled_centroidsY,
                scaled_y, scaled_x, per_centroid_labels)
    """
    # Create binary mask using the adapter
    binary_seg = adapter.create_binary_mask(segmentation, pixel_id)

    # Get pixel coordinates
    pixel_y, pixel_x = np.where(binary_seg)

    if len(pixel_y) == 0:
        del binary_seg
        return None, None, None, None, None, None

    # Scale pixel coordinates to registration space
    scaled_y, scaled_x = pixel_y * y_scale, pixel_x * x_scale

    objects_info = adapter.extract_objects(segmentation, binary_seg, min_area=object_cutoff)

    if len(objects_info) == 0:
        return None, None, None, scaled_y, scaled_x, None

    centroids = []
    per_centroid_labels = []
    atlas_h, atlas_w = atlas_map.shape
    seg_height, seg_width = segmentation.shape[:2]

    # Map directly from segmentation pixel coords to atlas map coords.
    # Equivalent to the former atlas_at_original_resolution=True code path.
    seg_to_atlas_y = atlas_h / seg_height
    seg_to_atlas_x = atlas_w / seg_width

    for obj in objects_info:
        centroids.append(obj.centroid)
        # Map object pixel coords directly to atlas map resolution
        assignment_y = obj.coords[:, 0] * seg_to_atlas_y
        assignment_x = obj.coords[:, 1] * seg_to_atlas_x

        # Check bounds
        iy = np.rint(assignment_y).astype(np.int32, copy=False)
        ix = np.rint(assignment_x).astype(np.int32, copy=False)
        valid_mask = (
            (iy >= 0)
            & (iy < atlas_h)
            & (ix >= 0)
            & (ix < atlas_w)
        )

        if not np.any(valid_mask):
            per_centroid_labels.append(0)
            continue

        # Assign region based on majority vote of pixels in object
        pixel_labels = atlas_map[iy[valid_mask], ix[valid_mask]]
        pixel_labels = np.asarray(pixel_labels, dtype=np.int64)
        if pixel_labels.size == 0:
            per_centroid_labels.append(0)
            continue
        pixel_labels = pixel_labels[pixel_labels >= 0]
        if pixel_labels.size == 0:
            per_centroid_labels.append(0)
            continue
        counts = np.bincount(pixel_labels)
        per_centroid_labels.append(int(np.argmax(counts)))

    centroids = np.array(centroids)
    per_centroid_labels = np.array(per_centroid_labels, dtype=np.int64)

    scaled_centroidsY, scaled_centroidsX = (
        centroids[:, 0] * y_scale, centroids[:, 1] * x_scale
    )

    return (
        centroids,
        scaled_centroidsX,
        scaled_centroidsY,
        scaled_y,
        scaled_x,
        per_centroid_labels,
    )

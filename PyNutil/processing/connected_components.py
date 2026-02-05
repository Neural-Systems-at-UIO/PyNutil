"""Connected component analysis for segmentation images.

This module provides functions for detecting and analyzing connected
components in binary segmentation masks, including centroid extraction
and region assignment.
"""

import numpy as np
import cv2

from .utils import scale_positions


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
    order = np.argsort(comp_ids, kind="stable")
    ys = ys[order]
    xs = xs[order]
    comp_ids = comp_ids[order]

    unique_ids, start_idx = np.unique(comp_ids, return_index=True)
    end_idx = np.r_[start_idx[1:], comp_ids.size]

    props = []
    for comp_id, start, end in zip(unique_ids, start_idx, end_idx):
        # comp_id is >= 1 (0 is background)
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        cx, cy = centroids_xy[comp_id]
        coords = np.column_stack((ys[start:end], xs[start:end]))
        props.append(
            {"area": area, "centroid": (float(cy), float(cx)), "coords": coords}
        )
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

    # Sort by label ID to group pixels of the same object
    order = np.argsort(labels, kind="stable")
    ys = ys[order]
    xs = xs[order]
    labels = labels[order]

    # Find boundaries of each label group
    unique_ids, start_idx = np.unique(labels, return_index=True)
    end_idx = np.r_[start_idx[1:], labels.size]

    props = []
    for label_id, start, end in zip(unique_ids, start_idx, end_idx):
        comp_ys = ys[start:end]
        comp_xs = xs[start:end]
        area = len(comp_ys)
        centroid = (np.mean(comp_ys), np.mean(comp_xs))
        coords = np.column_stack((comp_ys, comp_xs))
        props.append({"area": area, "centroid": centroid, "coords": coords})
    return props


def get_centroids_and_area(segmentation, pixel_cut_off=0):
    """Retrieve centroids, areas, and pixel coordinates of labeled regions.

    Args:
        segmentation: Binary segmentation array.
        pixel_cut_off: Minimum object size threshold.

    Returns:
        tuple: (centroids, area, coords) of retained objects.
    """
    labels_info = connected_components_props(
        segmentation.astype(bool, copy=False), connectivity=4
    )
    labels_info = [label for label in labels_info if label["area"] > pixel_cut_off]
    centroids = np.array([label["centroid"] for label in labels_info])
    area = np.array([label["area"] for label in labels_info])
    coords = np.array([label["coords"] for label in labels_info], dtype=object)
    return centroids, area, coords


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
    from .adapters import SegmentationAdapterRegistry

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
    objects_info = adapter.extract_objects(segmentation, binary_seg, min_area=object_cutoff)

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
            # Scale COORDINATES down to atlas size, instead of scaling atlas up
            atlas_height, atlas_width = atlas_map.shape
            assignment_y = scaled_obj_y * (atlas_height / reg_height)
            assignment_x = scaled_obj_x * (atlas_width / reg_width)
            atlas_bounds_height, atlas_bounds_width = atlas_height, atlas_width
        else:
            assignment_y, assignment_x = scaled_obj_y, scaled_obj_x
            atlas_bounds_height, atlas_bounds_width = (
                atlas_map.shape[0],
                atlas_map.shape[1],
            )

        # Check bounds
        valid_mask = (
            (np.round(assignment_y).astype(int) >= 0)
            & (np.round(assignment_y).astype(int) < atlas_bounds_height)
            & (np.round(assignment_x).astype(int) >= 0)
            & (np.round(assignment_x).astype(int) < atlas_bounds_width)
        )

        if not np.any(valid_mask):
            per_centroid_labels.append(0)
            continue

        # Assign Region based on majority vote of pixels in object
        pixel_labels = atlas_map[
            np.round(assignment_y[valid_mask]).astype(int),
            np.round(assignment_x[valid_mask]).astype(int),
        ]
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

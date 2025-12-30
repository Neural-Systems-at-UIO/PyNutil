import numpy as np
import pandas as pd
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from ..io.read_and_write import load_quint_json, load_segmentation
from .counting_and_load import flat_to_dataframe, rescale_image, load_image
from .generate_target_slice import generate_target_slice
import cv2
import threading
from .transformations import (
    transform_points_to_atlas_space,
    transform_to_registration,
    get_transformed_coordinates,
    transform_to_atlas_space,
)
from .image_loaders import detect_pixel_id
from .transform import get_region_areas, get_triangulation
from .aggregator import build_region_intensity_dataframe
from .utils import (
    get_flat_files,
    get_segmentations,
    number_sections,
    find_matching_pixels,
    scale_positions,
    process_results,
    get_current_flat_file,
    start_and_join_threads,
    convert_to_intensity,
    update_spacing,
    create_damage_mask,
)


def _connected_components_props(binary_mask: np.ndarray, *, connectivity: int = 4):
    """Return connected-component properties for a boolean 2D mask.

    Uses OpenCV so we can avoid scikit-image dependency.

    Returns a list of dicts with:
        - area (int)
        - centroid (tuple[float, float])  (y, x)
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


def _labeled_image_props(label_image: np.ndarray):


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


def get_objects_and_assign_regions_optimized(
    segmentation,
    pixel_id,
    atlas_map,
    y_scale,
    x_scale,
    object_cutoff=0,
    atlas_at_original_resolution=False,
    reg_height=None,
    reg_width=None,
    cellpose=False,
):
    """Single-pass object detection, pixel extraction, and region assignment."""

    # Memory-efficient binary segmentation: process channels separately
    if cellpose:
        if segmentation.ndim == 2:
            binary_seg = segmentation != 0
        else:
            binary_seg = segmentation[:, :, 0] != 0
    else:
        if segmentation.ndim == 2:
            binary_seg = segmentation == pixel_id[0]
        else:
            binary_seg = segmentation[:, :, 0] == pixel_id[0]
            if segmentation.shape[2] > 1:
                binary_seg &= segmentation[:, :, 1] == pixel_id[1]
            if segmentation.shape[2] > 2:
                binary_seg &= segmentation[:, :, 2] == pixel_id[2]

    # Get pixel coordinates
    pixel_y, pixel_x = np.where(binary_seg)

    if len(pixel_y) == 0:
        del binary_seg
        return None, None, None, None, None, None

    # Scale pixel coordinates to registration space
    scaled_y, scaled_x = scale_positions(pixel_y, pixel_x, y_scale, x_scale)

    # Object Detection
    if cellpose:
        objects_info = _labeled_image_props(segmentation)
    else:
        # connectivity=4 matches skimage.measure.label(..., connectivity=1) for 2D.
        objects_info = _connected_components_props(binary_seg, connectivity=4)
    objects_info = [obj for obj in objects_info if obj["area"] > object_cutoff]

    if len(objects_info) == 0:
        return None, None, None, scaled_y, scaled_x, None

    centroids = []
    per_centroid_labels = []

    for obj in objects_info:
        centroids.append(obj["centroid"])
        # Scale object coords
        scaled_obj_y, scaled_obj_x = scale_positions(
            obj["coords"][:, 0], obj["coords"][:, 1], y_scale, x_scale
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


def get_centroids_and_area(segmentation, pixel_cut_off=0):
    """
    Retrieves centroids, areas, and pixel coordinates of labeled regions.

    Args:
        segmentation (ndarray): Binary segmentation array.
        pixel_cut_off (int, optional): Minimum object size threshold.

    Returns:
        tuple: (centroids, area, coords) of retained objects.
    """
    labels_info = _connected_components_props(
        segmentation.astype(bool, copy=False), connectivity=4
    )
    labels_info = [label for label in labels_info if label["area"] > pixel_cut_off]
    centroids = np.array([label["centroid"] for label in labels_info])
    area = np.array([label["area"] for label in labels_info])
    coords = np.array([label["coords"] for label in labels_info], dtype=object)
    return centroids, area, coords


def folder_to_atlas_space(
    folder,
    quint_alignment,
    atlas_labels,
    pixel_id=[0, 0, 0],
    non_linear=True,
    object_cutoff=0,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    apply_damage_mask=True,
    cellpose=False,
):
    """
    Processes all segmentation files in a folder, mapping each one to atlas space.

    Args:
        folder (str): Path to segmentation files.
        quint_alignment (str): Path to alignment JSON.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        pixel_id (list, optional): Pixel color to match.
        non_linear (bool, optional): Apply non-linear transform.
        object_cutoff (int, optional): Minimum object size.
        atlas_volume (ndarray, optional): Atlas volume data.
        hemi_map (ndarray, optional): Hemisphere mask data.
        use_flat (bool, optional): If True, load flat files.
        apply_damage_mask (bool, optional): If True, apply damage mask.
        cellpose (bool, optional): If True, the segmentation files are assumed to be Cellpose output.

    Returns:
        tuple: Various arrays and lists containing transformed coordinates and labels.
    """
    quint_json = load_quint_json(quint_alignment)
    slices = quint_json["slices"]
    if apply_damage_mask and "gridspacing" in quint_json:
        gridspacing = quint_json["gridspacing"]
    else:
        gridspacing = None
    if not apply_damage_mask:
        for slice in slices:
            if "grid" in slice:
                slice.pop("grid")
    segmentations = get_segmentations(folder)
    flat_files, flat_file_nrs = get_flat_files(folder, use_flat)
    region_areas_list = [
        pd.DataFrame(
            {
                "idx": [],
                "name": [],
                "r": [],
                "g": [],
                "b": [],
                "region_area": [],
                "pixel_count": [],
                "object_count": [],
                "area_fraction": [],
            }
        )
        for _ in range(len(segmentations))
    ]
    points_list = [np.array([], dtype=np.float64) for _ in range(len(segmentations))]
    points_labels = [np.array([], dtype=np.int64) for _ in range(len(segmentations))]
    centroids_list = [np.array([], dtype=np.float64) for _ in range(len(segmentations))]
    centroids_labels = [np.array([], dtype=np.int64) for _ in range(len(segmentations))]
    per_point_undamaged_list = [np.array([], dtype=bool) for _ in range(len(segmentations))]
    per_centroid_undamaged_list = [
        np.array([], dtype=bool) for _ in range(len(segmentations))
    ]
    points_hemi_labels = [np.array([], dtype=np.int64) for _ in range(len(segmentations))]
    centroids_hemi_labels = [
        np.array([], dtype=np.int64) for _ in range(len(segmentations))
    ]

    if len(segmentations) > 0:
        max_workers = min(32, len(segmentations), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for segmentation_path, index in zip(segmentations, range(len(segmentations))):
                seg_nr = int(number_sections([segmentation_path])[0])
                current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
                if len(current_slice_index[0]) == 0:
                    print("segmentation file does not exist in alignment json:")
                    print(segmentation_path)
                    continue

                current_slice = slices[current_slice_index[0][0]]
                if current_slice["anchoring"] == []:
                    continue

                current_flat = get_current_flat_file(
                    seg_nr, flat_files, flat_file_nrs, use_flat
                )

                futures.append(
                    executor.submit(
                        segmentation_to_atlas_space,
                        current_slice,
                        segmentation_path,
                        atlas_labels,
                        current_flat,
                        pixel_id,
                        non_linear,
                        points_list,
                        centroids_list,
                        points_labels,
                        centroids_labels,
                        region_areas_list,
                        per_point_undamaged_list,
                        per_centroid_undamaged_list,
                        points_hemi_labels,
                        centroids_hemi_labels,
                        index,
                        object_cutoff,
                        atlas_volume,
                        hemi_map,
                        use_flat,
                        gridspacing,
                        cellpose=cellpose,
                    )
                )

            # Ensure exceptions from worker threads get raised here.
            for f in futures:
                f.result()
    (
        points,
        centroids,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        points_len,
        centroids_len,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
    ) = process_results(
        points_list,
        centroids_list,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
    )
    return (
        points,
        centroids,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        region_areas_list,
        points_len,
        centroids_len,
        segmentations,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
    )


def folder_to_atlas_space_intensity(
    folder,
    quint_alignment,
    atlas_labels,
    intensity_channel="grayscale",
    non_linear=True,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    apply_damage_mask=True,
    min_intensity=None,
    max_intensity=None,
):
    """
    Processes all images in a folder, mapping each one to atlas space and extracting intensity.

    Args:
        folder (str): Path to image files.
        quint_alignment (str): Path to alignment JSON.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        intensity_channel (str, optional): Channel to use for intensity.
        non_linear (bool, optional): Apply non-linear transform.
        atlas_volume (ndarray, optional): Atlas volume data.
        hemi_map (ndarray, optional): Hemisphere mask data.
        use_flat (bool, optional): If True, load flat files.
        apply_damage_mask (bool, optional): If True, apply damage mask.
        min_intensity (int, optional): Minimum intensity value to include.
        max_intensity (int, optional): Maximum intensity value to include.

    Returns:
        tuple: (region_intensities_list, images, centroids, centroids_labels, centroids_hemi_labels, centroids_len, centroids_intensities)
    """
    quint_json = load_quint_json(quint_alignment)
    slices = quint_json["slices"]
    if apply_damage_mask and "gridspacing" in quint_json:
        gridspacing = quint_json["gridspacing"]
    else:
        gridspacing = None
    if not apply_damage_mask:
        for slice in slices:
            if "grid" in slice:
                slice.pop("grid")
    images = get_segmentations(folder)
    flat_files, flat_file_nrs = get_flat_files(folder, use_flat)

    region_intensities_list = [None] * len(images)
    centroids_list = [None] * len(images)
    centroids_labels_list = [None] * len(images)
    centroids_hemi_labels_list = [None] * len(images)
    centroids_intensities_list = [None] * len(images)
    centroids_len = [0] * len(images)

    if len(images) > 0:
        max_workers = min(32, len(images), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for image_path, index in zip(images, range(len(images))):
                seg_nr = int(number_sections([image_path])[0])
                current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
                if len(current_slice_index[0]) == 0:
                    print(f"image file does not exist in alignment json: {image_path}")
                    continue

                current_slice = slices[current_slice_index[0][0]]
                if current_slice["anchoring"] == []:
                    continue

                current_flat = get_current_flat_file(
                    seg_nr, flat_files, flat_file_nrs, use_flat
                )

                futures.append(
                    executor.submit(
                        segmentation_to_atlas_space_intensity,
                        current_slice,
                        image_path,
                        atlas_labels,
                        intensity_channel,
                        current_flat,
                        non_linear,
                        region_intensities_list,
                        index,
                        atlas_volume,
                        hemi_map,
                        use_flat,
                        gridspacing,
                        centroids_list,
                        centroids_labels_list,
                        centroids_hemi_labels_list,
                        centroids_intensities_list,
                        centroids_len,
                        min_intensity=min_intensity,
                        max_intensity=max_intensity,
                    )
                )

            # Ensure exceptions from worker threads get raised here.
            for f in futures:
                f.result()

    # Concatenate results
    all_centroids = np.concatenate([c for c in centroids_list if c is not None]) if any(c is not None for c in centroids_list) else None
    all_labels = np.concatenate([l for l in centroids_labels_list if l is not None]) if any(l is not None for l in centroids_labels_list) else None
    all_hemi = np.concatenate([h for h in centroids_hemi_labels_list if h is not None]) if any(h is not None for h in centroids_hemi_labels_list) else None
    all_intensities = np.concatenate([i for i in centroids_intensities_list if i is not None]) if any(i is not None for i in centroids_intensities_list) else None

    return (
        region_intensities_list,
        images,
        all_centroids,
        all_labels,
        all_hemi,
        centroids_len,
        all_intensities,
    )


def segmentation_to_atlas_space_intensity(
    slice_dict,
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
    grid_spacing=None,
    centroids_list=None,
    centroids_labels_list=None,
    centroids_hemi_labels_list=None,
    centroids_intensities_list=None,
    centroids_len=None,
    min_intensity=None,
    max_intensity=None,
):
    """
    Transforms a single image file into atlas space and extracts intensity.
    """
    image = load_segmentation(image_path)
    intensity = convert_to_intensity(image, intensity_channel)

    # Apply intensity filters if specified
    if min_intensity is not None:
        intensity[intensity < min_intensity] = 0
    if max_intensity is not None:
        intensity[intensity > max_intensity] = 0

    reg_height, reg_width = slice_dict["height"], slice_dict["width"]
    triangulation = get_triangulation(slice_dict, reg_width, reg_height, non_linear)

    if "grid" in slice_dict:
        damage_mask = create_damage_mask(slice_dict, grid_spacing)
    else:
        damage_mask = None

    if hemi_map is not None:
        hemi_mask = generate_target_slice(slice_dict["anchoring"], hemi_map)
        if hemi_mask.shape != (reg_height, reg_width):
            hemi_mask = cv2.resize(
                hemi_mask.astype(np.uint8),
                (reg_width, reg_height),
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        hemi_mask = None

    region_areas, atlas_map = get_region_areas(
        use_flat,
        atlas_labels,
        flat_file_atlas,
        image.shape[1],
        image.shape[0],
        slice_dict,
        atlas_volume,
        hemi_mask,
        triangulation,
        damage_mask,
    )
    atlas_map = atlas_map.astype(np.int64)
    if atlas_map.shape != (reg_height, reg_width):
        atlas_map = cv2.resize(
            atlas_map.astype(np.float32),
            (reg_width, reg_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

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
        damage_mask_resized = cv2.resize(
            damage_mask.astype(np.uint8),
            (reg_width, reg_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        intensity_resized[~damage_mask_resized] = 0
        atlas_map[~damage_mask_resized] = 0

    df = build_region_intensity_dataframe(
        atlas_map=atlas_map,
        intensity_resized=intensity_resized,
        atlas_labels=atlas_labels,
        region_areas=region_areas,
        hemi_mask=hemi_mask,
        damage_mask_resized=damage_mask_resized if damage_mask is not None else None,
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
            slice_dict["anchoring"],
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


def create_threads(
    segmentations,
    slices,
    flat_files,
    flat_file_nrs,
    atlas_labels,
    pixel_id,
    non_linear,
    points_list,
    centroids_list,
    centroids_labels,
    points_labels,
    region_areas_list,
    per_point_undamaged_list,
    per_centroid_undamaged_list,
    point_hemi_labels,
    centroid_hemi_labels,
    object_cutoff,
    atlas_volume,
    hemi_map,
    use_flat,
    gridspacing,
    cellpose=False,
):
    """
    Creates threads to transform each segmentation into atlas space.

    Args:
        segmentations (list): Paths to segmentation files.
        slices (list): Slice metadata from alignment JSON.
        flat_files (list): Flat file paths for optional flat maps.
        flat_file_nrs (list): Numeric indices for flat files.
        atlas_labels (DataFrame): Atlas labels.
        pixel_id (list): Pixel color defined as [R, G, B].
        non_linear (bool): Enable non-linear transformation if True.
        points_list (list): Stores point coordinates per segmentation.
        centroids_list (list): Stores centroid coordinates per segmentation.
        centroids_labels (list): Stores labels for each centroid array.
        points_labels (list): Stores labels for each point array.
        region_areas_list (list): Stores region area data per segmentation.
        per_point_undamaged_list (list): Track undamaged points per segmentation.
        per_centroid_undamaged_list (list): Track undamaged centroids per segmentation.
        point_hemi_labels (list): Hemisphere labels for points.
        centroid_hemi_labels (list): Hemisphere labels for centroids.
        object_cutoff (int): Minimum object size threshold.
        atlas_volume (ndarray): 3D atlas volume (optional).
        hemi_map (ndarray): Hemisphere mask (optional).
        use_flat (bool): Use flat files if True.
        gridspacing (int): Spacing value from alignment data.
        cellpose (bool): If True, the segmentation files are assumed to be Cellpose output.

    Returns:
        list: A list of threads for parallel execution.
    """
    threads = []
    for segmentation_path, index in zip(segmentations, range(len(segmentations))):
        seg_nr = int(number_sections([segmentation_path])[0])
        current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
        if len(current_slice_index[0]) == 0:
            print("segmentation file does not exist in alignment json:")
            print(segmentation_path)
            continue
        current_slice = slices[current_slice_index[0][0]]
        if current_slice["anchoring"] == []:
            continue
        current_flat = get_current_flat_file(
            seg_nr, flat_files, flat_file_nrs, use_flat
        )

        x = threading.Thread(
            target=segmentation_to_atlas_space,
            args=(
                current_slice,
                segmentation_path,
                atlas_labels,
                current_flat,
                pixel_id,
                non_linear,
                points_list,
                centroids_list,
                points_labels,
                centroids_labels,
                region_areas_list,
                per_point_undamaged_list,
                per_centroid_undamaged_list,
                point_hemi_labels,
                centroid_hemi_labels,
                index,
                object_cutoff,
                atlas_volume,
                hemi_map,
                use_flat,
                gridspacing,
                cellpose,
            ),
        )
        threads.append(x)
    return threads


def detect_pixel_id(segmentation: np.ndarray):
    """
    Infers pixel color from the first non-background region.

    Args:
        segmentation (ndarray): Segmentation array.

    Returns:
        ndarray: Identified pixel color (RGB or scalar).
    """
    if segmentation.ndim == 2:
        # For 2D images, find the first non-zero value
        non_zero = segmentation[segmentation != 0]
        if non_zero.size > 0:
            pixel_id = [int(non_zero[0])]
        else:
            pixel_id = [255]
    else:
        segmentation_no_background = segmentation[~np.all(segmentation == 0, axis=2)]
        if segmentation_no_background.size > 0:
            pixel_id = segmentation_no_background[0]
        else:
            pixel_id = [255, 255, 255]
    # print("detected pixel_id: ", pixel_id)
    return pixel_id


def segmentation_to_atlas_space(
    slice_dict,
    segmentation_path,
    atlas_labels,
    flat_file_atlas=None,
    pixel_id="auto",
    non_linear=True,
    points_list=None,
    centroids_list=None,
    points_labels=None,
    centroids_labels=None,
    region_areas_list=None,
    per_point_undamaged_list=None,
    per_centroid_undamaged_list=None,
    points_hemi_labels=None,
    centroids_hemi_labels=None,
    index=None,
    object_cutoff=0,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    grid_spacing=None,
    cellpose=False,
):
    """
    Transforms a single segmentation file into atlas space.

    Args:
        slice_dict (dict): Slice information from alignment JSON.
        segmentation_path (str): Path to the segmentation file.
        atlas_labels (DataFrame): Atlas labels.
        flat_file_atlas (str, optional): Path to flat atlas, if using flat files.
        pixel_id (str or list, optional): Pixel color or 'auto'.
        non_linear (bool, optional): Use non-linear transforms if True.
        points_list (list, optional): Storage for transformed point coordinates.
        centroids_list (list, optional): Storage for transformed centroid coordinates.
        points_labels (list, optional): Storage for assigned point labels.
        centroids_labels (list, optional): Storage for assigned centroid labels.
        region_areas_list (list, optional): Storage for region area data.
        per_point_undamaged_list (list, optional): Track undamaged points.
        per_centroid_undamaged_list (list, optional): Track undamaged centroids.
        points_hemi_labels (list, optional): Hemisphere labels for points.
        centroids_hemi_labels (list, optional): Hemisphere labels for centroids.
        index (int, optional): Index in the lists.
        object_cutoff (int, optional): Minimum object size.
        atlas_volume (ndarray, optional): 3D atlas volume.
        hemi_map (ndarray, optional): Hemisphere mask.
        use_flat (bool, optional): Indicates use of flat files.
        grid_spacing (int, optional): Spacing value for damage mask.
        cellpose (bool, optional): If True, the segmentation files are assumed to be Cellpose output.

    Returns:
        None
    """
    segmentation = load_segmentation(segmentation_path)
    if pixel_id == "auto" and not cellpose:
        pixel_id = detect_pixel_id(segmentation)
    seg_height, seg_width = segmentation.shape[:2]
    reg_height, reg_width = slice_dict["height"], slice_dict["width"]
    triangulation = get_triangulation(slice_dict, reg_width, reg_height, non_linear)
    if "grid" in slice_dict:
        damage_mask = create_damage_mask(slice_dict, grid_spacing)
    else:
        damage_mask = None
    if hemi_map is not None:
        hemi_mask = generate_target_slice(slice_dict["anchoring"], hemi_map)
    else:
        hemi_mask = None
    region_areas, atlas_map = get_region_areas(
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
    ) = get_objects_and_assign_regions_optimized(
        segmentation,
        pixel_id,
        atlas_map,
        y_scale,
        x_scale,
        object_cutoff=object_cutoff,
        atlas_at_original_resolution=True,
        reg_height=reg_height,
        reg_width=reg_width,
        cellpose=cellpose,
    )

    if scaled_y is None or scaled_x is None:
        points_list[index] = np.array([], dtype=np.float64)
        centroids_list[index] = np.array([], dtype=np.float64)
        region_areas_list[index] = region_areas
        centroids_labels[index] = np.array([], dtype=np.int64)
        per_centroid_undamaged_list[index] = np.array([], dtype=bool)
        points_labels[index] = np.array([], dtype=np.int64)
        per_point_undamaged_list[index] = np.array([], dtype=bool)
        points_hemi_labels[index] = np.array([], dtype=np.int64)
        centroids_hemi_labels[index] = np.array([], dtype=np.int64)
        gc.collect()
        return

    # Assign per-pixel labels using atlas at original resolution (scale coords down)
    atlas_height, atlas_width = atlas_map.shape
    assignment_y = scaled_y * (atlas_height / reg_height)
    assignment_x = scaled_x * (atlas_width / reg_width)
    ay = np.round(assignment_y).astype(int)
    ax = np.round(assignment_x).astype(int)
    valid = (ay >= 0) & (ay < atlas_height) & (ax >= 0) & (ax < atlas_width)
    per_point_labels = np.zeros_like(ay, dtype=atlas_map.dtype)
    if np.any(valid):
        per_point_labels[valid] = atlas_map[ay[valid], ax[valid]]
    if damage_mask is not None:
        damage_mask = cv2.resize(
            damage_mask.astype(np.uint8),
            (reg_width, reg_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
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
        # Assign hemisphere labels in atlas space (no atlas upscaling)
        per_point_hemi = np.zeros_like(ay, dtype=hemi_mask.dtype)
        if np.any(valid):
            per_point_hemi[valid] = hemi_mask[ay[valid], ax[valid]]

        if scaled_centroidsX is not None and scaled_centroidsY is not None:
            c_assignment_y = scaled_centroidsY * (atlas_height / reg_height)
            c_assignment_x = scaled_centroidsX * (atlas_width / reg_width)
            cy = np.round(c_assignment_y).astype(int)
            cx = np.round(c_assignment_x).astype(int)
            c_valid = (cy >= 0) & (cy < atlas_height) & (cx >= 0) & (cx < atlas_width)
            per_centroid_hemi = np.zeros_like(cy, dtype=hemi_mask.dtype)
            if np.any(c_valid):
                per_centroid_hemi[c_valid] = hemi_mask[cy[c_valid], cx[c_valid]]
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
        slice_dict,
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
        triangulation,
    )
    points, centroids = transform_points_to_atlas_space(
        slice_dict,
        new_x,
        new_y,
        centroids_new_x,
        centroids_new_y,
        reg_height,
        reg_width,
    )
    points_list[index] = np.array(points if points is not None else [])
    centroids_list[index] = np.array(centroids if centroids is not None else [])
    region_areas_list[index] = region_areas
    centroids_labels[index] = np.array(
        per_centroid_labels if centroids is not None else []
    )
    per_centroid_undamaged_list[index] = np.array(
        per_centroid_undamaged if centroids is not None else []
    )
    points_labels[index] = np.array(per_point_labels if points is not None else [])
    per_point_undamaged_list[index] = np.array(
        per_point_undamaged if points is not None else []
    )
    points_hemi_labels[index] = np.array(per_point_hemi if points is not None else [])
    centroids_hemi_labels[index] = np.array(
        per_centroid_hemi if points is not None else []
    )

    gc.collect()


def get_centroids(segmentation, pixel_id, y_scale, x_scale, object_cutoff=0):
    """
    Finds object centroids for a given pixel color and applies scaling.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_id (int): Pixel color to match.
        y_scale (float): Vertical scaling factor.
        x_scale (float): Horizontal scaling factor.
        object_cutoff (int, optional): Minimum object size.

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
    """
    Retrieves pixel coordinates for a specified color and scales them.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_id (int): Pixel color to match.
        y_scale (float): Vertical scaling factor.
        x_scale (float): Horizontal scaling factor.

    Returns:
        tuple: (scaled_y, scaled_x)
    """
    id_pixels = find_matching_pixels(segmentation, pixel_id)
    if len(id_pixels[0]) == 0:
        return None, None
    scaled_y, scaled_x = scale_positions(id_pixels[0], id_pixels[1], y_scale, x_scale)
    return scaled_y, scaled_x

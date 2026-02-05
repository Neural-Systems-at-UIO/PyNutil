"""Batch processing for folder-level atlas space transformation.

This module contains functions for processing all segmentation files
in a folder, mapping each one to atlas space using parallel execution.
"""

import os
import threading

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .adapters import load_registration
from .section_processor import (
    segmentation_to_atlas_space,
    segmentation_to_atlas_space_intensity,
)
from .utils import (
    get_flat_files,
    get_segmentations,
    number_sections,
    process_results,
    get_current_flat_file,
)


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
    segmentation_format="binary",
):
    """Process all segmentation files in a folder, mapping each to atlas space.

    Args:
        folder: Path to segmentation files.
        quint_alignment: Path to alignment JSON.
        atlas_labels: DataFrame with atlas labels.
        pixel_id: Pixel color to match.
        non_linear: Apply non-linear transform.
        object_cutoff: Minimum object size.
        atlas_volume: Atlas volume data.
        hemi_map: Hemisphere mask data.
        use_flat: If True, load flat files.
        apply_damage_mask: If True, apply damage mask.
        segmentation_format: Format name ("binary" or "cellpose").

    Returns:
        tuple: (points, centroids, points_labels, centroids_labels,
                points_hemi_labels, centroids_hemi_labels, region_areas_list,
                points_len, centroids_len, segmentations,
                per_point_undamaged_list, per_centroid_undamaged_list)
    """
    # Load registration using the modular adapter system
    registration = load_registration(
        quint_alignment,
        apply_deformation=non_linear,
        apply_damage=apply_damage_mask,
    )
    slices_by_nr = {s.section_number: s for s in registration.slices}

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
                slice_info = slices_by_nr.get(seg_nr)
                if slice_info is None:
                    print("segmentation file does not exist in alignment json:")
                    print(segmentation_path)
                    continue

                if not slice_info.anchoring:
                    continue

                current_flat = get_current_flat_file(
                    seg_nr, flat_files, flat_file_nrs, use_flat
                )

                futures.append(
                    executor.submit(
                        segmentation_to_atlas_space,
                        slice_info,
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
                        segmentation_format=segmentation_format,
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
    """Process all images in a folder, mapping each to atlas space with intensity.

    Args:
        folder: Path to image files.
        quint_alignment: Path to alignment JSON.
        atlas_labels: DataFrame with atlas labels.
        intensity_channel: Channel to use for intensity.
        non_linear: Apply non-linear transform.
        atlas_volume: Atlas volume data.
        hemi_map: Hemisphere mask data.
        use_flat: If True, load flat files.
        apply_damage_mask: If True, apply damage mask.
        min_intensity: Minimum intensity value to include.
        max_intensity: Maximum intensity value to include.

    Returns:
        tuple: (region_intensities_list, images, centroids, centroids_labels,
                centroids_hemi_labels, centroids_len, centroids_intensities)
    """
    # Load registration using the modular adapter system
    registration = load_registration(
        quint_alignment,
        apply_deformation=non_linear,
        apply_damage=apply_damage_mask,
    )
    slices_by_nr = {s.section_number: s for s in registration.slices}

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
                slice_info = slices_by_nr.get(seg_nr)
                if slice_info is None:
                    print(f"image file does not exist in alignment json: {image_path}")
                    continue

                if not slice_info.anchoring:
                    continue

                current_flat = get_current_flat_file(
                    seg_nr, flat_files, flat_file_nrs, use_flat
                )

                futures.append(
                    executor.submit(
                        segmentation_to_atlas_space_intensity,
                        slice_info,
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
    segmentation_format="binary",
):
    """Create threads to transform each segmentation into atlas space.

    Note: This function is deprecated. Use folder_to_atlas_space with
    ThreadPoolExecutor instead.

    Args:
        segmentations: Paths to segmentation files.
        slices: Slice metadata from alignment JSON.
        flat_files: Flat file paths for optional flat maps.
        flat_file_nrs: Numeric indices for flat files.
        atlas_labels: Atlas labels DataFrame.
        pixel_id: Pixel color [R, G, B].
        non_linear: Enable non-linear transformation.
        points_list: Stores point coordinates per segmentation.
        centroids_list: Stores centroid coordinates per segmentation.
        centroids_labels: Stores labels for each centroid array.
        points_labels: Stores labels for each point array.
        region_areas_list: Stores region area data per segmentation.
        per_point_undamaged_list: Track undamaged points.
        per_centroid_undamaged_list: Track undamaged centroids.
        point_hemi_labels: Hemisphere labels for points.
        centroid_hemi_labels: Hemisphere labels for centroids.
        object_cutoff: Minimum object size threshold.
        atlas_volume: 3D atlas volume.
        hemi_map: Hemisphere mask.
        use_flat: Use flat files if True.
        gridspacing: Spacing value from alignment data.
        segmentation_format: Format name ("binary" or "cellpose").

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
                segmentation_format,
            ),
        )
        threads.append(x)
    return threads

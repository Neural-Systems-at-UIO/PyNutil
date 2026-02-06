"""Batch processing for folder-level atlas space transformation.

This module contains functions for processing all segmentation files
in a folder, mapping each one to atlas space using parallel execution.
"""

import os

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from ...results import SectionResult
from ..adapters import load_registration
from .section_processor import (
    segmentation_to_atlas_space,
    segmentation_to_atlas_space_intensity,
)
from ..utils import (
    get_flat_files,
    get_segmentations,
    number_sections,
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

    # One SectionResult per segmentation file, defaulting to empty.
    results = [SectionResult.empty() for _ in range(len(segmentations))]

    if len(segmentations) > 0:
        max_workers = min(32, len(segmentations), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for index, segmentation_path in enumerate(segmentations):
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
                    (
                        index,
                        executor.submit(
                            segmentation_to_atlas_space,
                            slice_info,
                            segmentation_path,
                            atlas_labels,
                            current_flat,
                            pixel_id,
                            non_linear,
                            object_cutoff,
                            atlas_volume,
                            hemi_map,
                            use_flat,
                            segmentation_format=segmentation_format,
                        ),
                    )
                )

            for idx, future in futures:
                results[idx] = future.result()

    # ── Concatenate SectionResults ────────────────────────────────────
    def _concat(arrays):
        non_empty = [a for a in arrays if a is not None and len(a) > 0]
        return np.concatenate(non_empty) if non_empty else np.array([], dtype=np.float64)

    def _concat_int(arrays):
        non_empty = [a for a in arrays if a is not None and len(a) > 0]
        return np.concatenate(non_empty) if non_empty else np.array([], dtype=np.int64)

    def _concat_bool(arrays):
        non_empty = [a for a in arrays if a is not None and len(a) > 0]
        return np.concatenate(non_empty) if non_empty else np.array([], dtype=bool)

    points = _concat([r.points for r in results])
    centroids = _concat([r.centroids for r in results])
    points_labels = _concat_int([r.points_labels for r in results])
    centroids_labels = _concat_int([r.centroids_labels for r in results])
    points_hemi_labels = _concat_int([r.points_hemi_labels for r in results])
    centroids_hemi_labels = _concat_int([r.centroids_hemi_labels for r in results])
    per_point_undamaged = _concat_bool([r.per_point_undamaged for r in results])
    per_centroid_undamaged = _concat_bool([r.per_centroid_undamaged for r in results])

    points_len = [len(r.points) if r.points is not None else 0 for r in results]
    centroids_len = [len(r.centroids) if r.centroids is not None else 0 for r in results]
    region_areas_list = [r.region_areas for r in results]

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
        per_point_undamaged,
        per_centroid_undamaged,
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

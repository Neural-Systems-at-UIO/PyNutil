"""Batch processing for folder-level atlas space transformation.

This module contains functions for processing all segmentation files
in a folder, mapping each one to atlas space using parallel execution.
"""

import os

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from ...results import SectionResult, IntensitySectionResult
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


# ---------------------------------------------------------------------------
# Shared batch scaffold
# ---------------------------------------------------------------------------

def _run_batch(
    folder,
    quint_alignment,
    non_linear,
    apply_damage_mask,
    use_flat,
    empty_result_factory,
    submit_fn,
):
    """Generic batch scaffold shared by binary and intensity pipelines.

    Handles registration loading, file discovery, thread-pool setup,
    per-section looping, and futures collection.

    Args:
        folder: Path to segmentation / image files.
        quint_alignment: Path to alignment JSON.
        non_linear: Apply non-linear transform.
        apply_damage_mask: Apply damage mask.
        use_flat: Load flat files.
        empty_result_factory: Callable returning a default empty result.
        submit_fn: ``submit_fn(executor, index, slice_info, seg_path, current_flat)``
            — submits the per-section work to the executor and returns a Future.

    Returns:
        tuple: (segmentations, results) where *results* is a list parallel to
               *segmentations*, each element being the Future's result.
    """
    registration = load_registration(
        quint_alignment,
        apply_deformation=non_linear,
        apply_damage=apply_damage_mask,
    )
    slices_by_nr = {s.section_number: s for s in registration.slices}

    segmentations = get_segmentations(folder)
    flat_files, flat_file_nrs = get_flat_files(folder, use_flat)

    results = [empty_result_factory() for _ in range(len(segmentations))]

    if segmentations:
        max_workers = min(32, len(segmentations), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for index, seg_path in enumerate(segmentations):
                seg_nr = int(number_sections([seg_path])[0])
                slice_info = slices_by_nr.get(seg_nr)
                if slice_info is None:
                    print(f"segmentation file does not exist in alignment json: {seg_path}")
                    continue
                if not slice_info.anchoring:
                    continue

                current_flat = get_current_flat_file(
                    seg_nr, flat_files, flat_file_nrs, use_flat
                )
                futures.append(
                    (index, submit_fn(executor, index, slice_info, seg_path, current_flat))
                )

            for idx, future in futures:
                results[idx] = future.result()

    return segmentations, results


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

    def _submit(executor, index, slice_info, seg_path, current_flat):
        return executor.submit(
            segmentation_to_atlas_space,
            slice_info,
            seg_path,
            atlas_labels,
            current_flat,
            pixel_id,
            non_linear,
            object_cutoff,
            atlas_volume,
            hemi_map,
            use_flat,
            segmentation_format=segmentation_format,
        )

    segmentations, results = _run_batch(
        folder, quint_alignment, non_linear, apply_damage_mask, use_flat,
        SectionResult.empty, _submit,
    )

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

    # Viz lengths: undamaged-only 3D coordinates (for MeshView slicing)
    points_len = [len(r.points) if r.points is not None else 0 for r in results]
    centroids_len = [len(r.centroids) if r.centroids is not None else 0 for r in results]

    # Total lengths: all points including damaged (for counting / PerEntityArrays)
    total_points_len = [
        len(r.per_point_undamaged) if r.per_point_undamaged is not None else 0
        for r in results
    ]
    total_centroids_len = [
        len(r.per_centroid_undamaged) if r.per_centroid_undamaged is not None else 0
        for r in results
    ]

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
        total_points_len,
        total_centroids_len,
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

    def _submit(executor, index, slice_info, seg_path, current_flat):
        return executor.submit(
            segmentation_to_atlas_space_intensity,
            slice_info,
            seg_path,
            atlas_labels,
            intensity_channel,
            current_flat,
            non_linear,
            atlas_volume,
            hemi_map,
            use_flat,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
        )

    images, results = _run_batch(
        folder, quint_alignment, non_linear, apply_damage_mask, use_flat,
        IntensitySectionResult.empty, _submit,
    )

    # ── Concatenate IntensitySectionResults ────────────────────────────
    region_intensities_list = [r.region_intensities for r in results]

    def _concat(arrays):
        non_empty = [a for a in arrays if a is not None and len(a) > 0]
        return np.concatenate(non_empty) if non_empty else None

    all_centroids = _concat([r.points for r in results])
    all_labels = _concat([r.points_labels for r in results])
    all_hemi = _concat([r.points_hemi_labels for r in results])
    all_intensities = _concat([r.point_intensities for r in results])
    centroids_len = [r.num_points for r in results]

    return (
        region_intensities_list,
        images,
        all_centroids,
        all_labels,
        all_hemi,
        centroids_len,
        all_intensities,
    )

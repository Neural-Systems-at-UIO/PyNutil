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
                    print(
                        f"segmentation file does not exist in alignment json: {seg_path}"
                    )
                    continue
                if not slice_info.anchoring:
                    continue

                current_flat = get_current_flat_file(
                    seg_nr, flat_files, flat_file_nrs, use_flat
                )
                futures.append(
                    (
                        index,
                        submit_fn(executor, index, slice_info, seg_path, current_flat),
                    )
                )

            for idx, future in futures:
                results[idx] = future.result()

    return segmentations, results


# ---------------------------------------------------------------------------
# Concatenation helpers
# ---------------------------------------------------------------------------


def _concat_float(arrays):
    """Concatenate arrays, returning empty float64 array if all empty."""
    non_empty = [a for a in arrays if a is not None and len(a) > 0]
    return np.concatenate(non_empty) if non_empty else np.array([], dtype=np.float64)


def _concat_int(arrays):
    """Concatenate arrays, returning empty int64 array if all empty."""
    non_empty = [a for a in arrays if a is not None and len(a) > 0]
    return np.concatenate(non_empty) if non_empty else np.array([], dtype=np.int64)


def _concat_bool(arrays):
    """Concatenate arrays, returning empty bool array if all empty."""
    non_empty = [a for a in arrays if a is not None and len(a) > 0]
    return np.concatenate(non_empty) if non_empty else np.array([], dtype=bool)


def _concat_or_none(arrays):
    """Concatenate arrays, returning None if all empty (for intensity pipeline)."""
    non_empty = [a for a in arrays if a is not None and len(a) > 0]
    return np.concatenate(non_empty) if non_empty else None


def _safe_len(arr):
    """Return len(arr) if arr is not None, else 0."""
    return len(arr) if arr is not None else 0


def _unzip_section_results(results):
    """Unzip a list of SectionResults into per-field lists (single pass)."""
    pts, ctrs = [], []
    pts_lbl, ctrs_lbl = [], []
    pts_hemi, ctrs_hemi = [], []
    pt_undam, ct_undam = [], []
    pts_len, ctrs_len = [], []
    tot_pts_len, tot_ctrs_len = [], []
    areas = []

    for r in results:
        pts.append(r.points)
        ctrs.append(r.centroids)
        pts_lbl.append(r.points_labels)
        ctrs_lbl.append(r.centroids_labels)
        pts_hemi.append(r.points_hemi_labels)
        ctrs_hemi.append(r.centroids_hemi_labels)
        pt_undam.append(r.per_point_undamaged)
        ct_undam.append(r.per_centroid_undamaged)
        pts_len.append(_safe_len(r.points))
        ctrs_len.append(_safe_len(r.centroids))
        tot_pts_len.append(_safe_len(r.per_point_undamaged))
        tot_ctrs_len.append(_safe_len(r.per_centroid_undamaged))
        areas.append(r.region_areas)

    return (
        pts,
        ctrs,
        pts_lbl,
        ctrs_lbl,
        pts_hemi,
        ctrs_hemi,
        pt_undam,
        ct_undam,
        pts_len,
        ctrs_len,
        tot_pts_len,
        tot_ctrs_len,
        areas,
    )


def _collect_section_results(results):
    """Concatenate a list of SectionResults into flat arrays and length lists."""
    (
        pts,
        ctrs,
        pts_lbl,
        ctrs_lbl,
        pts_hemi,
        ctrs_hemi,
        pt_undam,
        ct_undam,
        pts_len,
        ctrs_len,
        tot_pts_len,
        tot_ctrs_len,
        areas,
    ) = _unzip_section_results(results)

    return (
        _concat_float(pts),
        _concat_float(ctrs),
        _concat_int(pts_lbl),
        _concat_int(ctrs_lbl),
        _concat_int(pts_hemi),
        _concat_int(ctrs_hemi),
        areas,
        pts_len,
        ctrs_len,
        _concat_bool(pt_undam),
        _concat_bool(ct_undam),
        tot_pts_len,
        tot_ctrs_len,
    )


# ---------------------------------------------------------------------------
# Binary pipeline
# ---------------------------------------------------------------------------


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
        folder,
        quint_alignment,
        non_linear,
        apply_damage_mask,
        use_flat,
        SectionResult.empty,
        _submit,
    )

    (
        points,
        centroids,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        region_areas_list,
        points_len,
        centroids_len,
        per_point_undamaged,
        per_centroid_undamaged,
        total_points_len,
        total_centroids_len,
    ) = _collect_section_results(results)

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
        folder,
        quint_alignment,
        non_linear,
        apply_damage_mask,
        use_flat,
        IntensitySectionResult.empty,
        _submit,
    )

    # ── Concatenate IntensitySectionResults ────────────────────────────
    region_intensities_list = [r.region_intensities for r in results]

    all_centroids = _concat_or_none([r.points for r in results])
    all_labels = _concat_or_none([r.points_labels for r in results])
    all_hemi = _concat_or_none([r.points_hemi_labels for r in results])
    all_intensities = _concat_or_none([r.point_intensities for r in results])
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

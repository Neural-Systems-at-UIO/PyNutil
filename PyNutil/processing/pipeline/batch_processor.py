"""Batch processing for folder-level atlas space transformation.

This module contains functions for processing all segmentation files
in a folder, mapping each one to atlas space using parallel execution.
"""

import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ...context import PipelineContext, SectionContext
from ...results import (
    SectionResult,
    IntensitySectionResult,
    ExtractionResult,
)
from ..adapters.base import RegistrationData
from ...results import AtlasData
from .section_processor import (
    segmentation_to_atlas_space,
    segmentation_to_atlas_space_intensity,
    coordinates_to_atlas_space,
)
from ..utils import (
    discover_image_files,
)
from ...io.loaders import number_sections


# ---------------------------------------------------------------------------
# Shared batch scaffold
# ---------------------------------------------------------------------------


def _run_batch_with_context(
    folder,
    registration: RegistrationData,
    pipeline_ctx: PipelineContext,
    empty_result_factory,
    processing_fn,
):
    """Generic batch scaffold using context objects.

    Handles file discovery, thread-pool setup, per-section looping,
    and futures collection.

    Args:
        folder: Path to segmentation / image files.
        registration: Pre-loaded registration data.
        pipeline_ctx: Immutable pipeline-wide state.
        empty_result_factory: Callable returning a default empty result.
        processing_fn: ``fn(p_ctx, s_ctx)`` — processes one section.

    Returns:
        tuple: (segmentations, results) where *results* is a list parallel to
               *segmentations*, each element being the Future's result.
    """
    slices_by_nr = {s.section_number: s for s in registration.slices}

    segmentations = discover_image_files(folder)

    # Discover flat files (only needed when use_flat is True)
    flat_files, flat_file_nrs = [], []
    if pipeline_ctx.use_flat:
        flat_dir = os.path.join(folder, "flat_files")
        flat_files = [
            os.path.join(flat_dir, name)
            for name in os.listdir(flat_dir)
            if name.endswith(".flat") or name.endswith(".seg")
        ]
        print(f"Found {len(flat_files)} flat files in folder {folder}")
        flat_file_nrs = [int(number_sections([ff])[0]) for ff in flat_files]

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

                current_flat = None
                if pipeline_ctx.use_flat:
                    idx_arr = [i for i, nr in enumerate(flat_file_nrs) if nr == seg_nr]
                    current_flat = flat_files[idx_arr[0]] if idx_arr else None
                section_ctx = SectionContext(
                    section_number=seg_nr,
                    slice_info=slice_info,
                    segmentation_path=seg_path,
                    flat_file_path=current_flat,
                )
                futures.append(
                    (
                        index,
                        executor.submit(processing_fn, pipeline_ctx, section_ctx),
                    )
                )

            for idx, future in futures:
                results[idx] = future.result()

    return segmentations, results


# ---------------------------------------------------------------------------
# Concatenation helpers
# ---------------------------------------------------------------------------


def _concat(arrays, *, dtype=None, none_if_empty=False):
    """Concatenate arrays with configurable dtype and empty-result behavior."""
    non_empty = [a for a in arrays if a is not None and len(a) > 0]
    if non_empty:
        result = np.concatenate(non_empty)
        # Only coerce dtype for numeric arrays; object arrays (e.g. hemi labels
        # that are [None, ...] when no hemisphere map is available) must be left
        # as-is so that downstream None-aware code still works correctly.
        if dtype is not None and result.dtype != object:
            return result.astype(dtype, copy=False)
        return result
    if none_if_empty:
        return None
    return np.array([], dtype=dtype)


def _collect_section_results(results):
    """Reduce section results into concatenated arrays and per-section lengths."""
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
        pts_len.append(len(r.points) if r.points is not None else 0)
        ctrs_len.append(len(r.centroids) if r.centroids is not None else 0)
        tot_pts_len.append(
            len(r.per_point_undamaged) if r.per_point_undamaged is not None else 0
        )
        tot_ctrs_len.append(
            len(r.per_centroid_undamaged)
            if r.per_centroid_undamaged is not None
            else 0
        )
        areas.append(r.region_areas)

    return (
        _concat(pts, dtype=np.float64),
        _concat(ctrs, dtype=np.float64),
        _concat(pts_lbl, dtype=np.int64),
        _concat(ctrs_lbl, dtype=np.int64),
        _concat(pts_hemi, dtype=np.int64),
        _concat(ctrs_hemi, dtype=np.int64),
        areas,
        pts_len,
        ctrs_len,
        _concat(pt_undam, dtype=bool),
        _concat(ct_undam, dtype=bool),
        tot_pts_len,
        tot_ctrs_len,
    )


# ---------------------------------------------------------------------------
# Binary pipeline
# ---------------------------------------------------------------------------


def folder_to_atlas_space(
    folder,
    registration: RegistrationData,
    atlas: AtlasData,
    pixel_id=[0, 0, 0],
    object_cutoff=0,
    use_flat=False,
    flat_label_path=None,
    segmentation_format="binary",
):
    """Process all segmentation files in a folder, mapping each to atlas space.

    Args:
        folder: Path to segmentation files.
        registration: Pre-loaded registration data.
        atlas: Atlas data bundle (volume, hemi_map, labels).
        pixel_id: Pixel color to match.
        object_cutoff: Minimum object size.
        use_flat: If True, load flat files.
        segmentation_format: Format name ("binary" or "cellpose").

    Returns:
        ExtractionResult: Structured extraction output.
    """
    pipeline_ctx = PipelineContext.from_format(
        segmentation_format=segmentation_format,
        atlas_labels=atlas.labels,
        atlas_volume=atlas.volume,
        hemi_map=atlas.hemi_map,
        non_linear=True,
        object_cutoff=object_cutoff,
        use_flat=use_flat,
        pixel_id=pixel_id,
        apply_damage_mask=True,
        flat_label_path=flat_label_path,
    )

    segmentations, results = _run_batch_with_context(
        folder,
        registration,
        pipeline_ctx,
        SectionResult.empty,
        segmentation_to_atlas_space,
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

    return ExtractionResult(
        pixel_points=points,
        centroids=centroids,
        points_labels=points_labels,
        centroids_labels=centroids_labels,
        points_hemi_labels=points_hemi_labels,
        centroids_hemi_labels=centroids_hemi_labels,
        region_areas_list=region_areas_list,
        points_len=points_len,
        centroids_len=centroids_len,
        segmentation_filenames=segmentations,
        per_point_undamaged=per_point_undamaged,
        per_centroid_undamaged=per_centroid_undamaged,
        total_points_len=total_points_len,
        total_centroids_len=total_centroids_len,
    )


def folder_to_atlas_space_intensity(
    folder,
    registration: RegistrationData,
    atlas: AtlasData,
    intensity_channel="grayscale",
    use_flat=False,
    flat_label_path=None,
    min_intensity=None,
    max_intensity=None,
):
    """Process all images in a folder, mapping each to atlas space with intensity.

    Args:
        folder: Path to image files.
        registration: Pre-loaded registration data.
        atlas: Atlas data bundle (volume, hemi_map, labels).
        intensity_channel: Channel to use for intensity.
        use_flat: If True, load flat files.
        min_intensity: Minimum intensity value to include.
        max_intensity: Maximum intensity value to include.

    Returns:
        ExtractionResult: Structured extraction output.
    """
    pipeline_ctx = PipelineContext.from_format(
        segmentation_format="binary",
        atlas_labels=atlas.labels,
        atlas_volume=atlas.volume,
        hemi_map=atlas.hemi_map,
        non_linear=True,
        object_cutoff=0,
        use_flat=use_flat,
        pixel_id=[0, 0, 0],
        apply_damage_mask=True,
        flat_label_path=flat_label_path,
        intensity_channel=intensity_channel,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
    )

    images, results = _run_batch_with_context(
        folder,
        registration,
        pipeline_ctx,
        IntensitySectionResult.empty,
        segmentation_to_atlas_space_intensity,
    )

    # ── Concatenate IntensitySectionResults ────────────────────────────
    region_intensities_list = [r.region_intensities for r in results]

    all_centroids = _concat([r.points for r in results], none_if_empty=True)
    all_labels = _concat([r.points_labels for r in results], none_if_empty=True)
    all_hemi = _concat([r.points_hemi_labels for r in results], none_if_empty=True)
    all_intensities = _concat(
        [r.point_intensities for r in results], none_if_empty=True
    )
    centroids_len = [r.num_points for r in results]

    return ExtractionResult(
        pixel_points=all_centroids,
        centroids=None,
        points_labels=all_labels,
        centroids_labels=None,
        points_hemi_labels=all_hemi,
        centroids_hemi_labels=None,
        region_areas_list=[],
        points_len=centroids_len,
        centroids_len=None,
        segmentation_filenames=images,
        per_point_undamaged=None,
        per_centroid_undamaged=None,
        total_points_len=centroids_len,
        total_centroids_len=None,
        region_intensities_list=region_intensities_list,
        point_intensities=all_intensities,
    )


# ---------------------------------------------------------------------------
# Coordinate pipeline
# ---------------------------------------------------------------------------


def file_to_atlas_space_coordinates(
    coordinate_file,
    registration: RegistrationData,
    atlas: AtlasData,
):
    """Process a coordinate CSV file, transforming points to atlas space.

    Loads coordinates from a CSV, groups by section number, and applies
    the full transformation pipeline (scaling, deformation, anchoring)
    to each section's coordinates.

    Args:
        coordinate_file: Path to the coordinate CSV file.
        registration: Pre-loaded registration data.
        atlas: Atlas data bundle (volume, hemi_map, labels).

    Returns:
        ExtractionResult: Structured extraction output.
    """
    from ...io.loaders import load_coordinate_file

    coord_df = load_coordinate_file(coordinate_file)

    slices_by_nr = {s.section_number: s for s in registration.slices}

    # Build a minimal PipelineContext (no segmentation adapter needed for coordinates)
    pipeline_ctx = PipelineContext.from_format(
        segmentation_format="binary",
        atlas_labels=atlas.labels,
        atlas_volume=atlas.volume,
        hemi_map=atlas.hemi_map,
        non_linear=True,
        object_cutoff=0,
        use_flat=False,
        pixel_id=[0, 0, 0],
        apply_damage_mask=True,
    )

    results = []
    for section_nr, group in coord_df.groupby("section number"):
        section_nr = int(section_nr)
        slice_info = slices_by_nr.get(section_nr)
        if slice_info is None:
            print(
                f"Section {section_nr} from coordinate file not found in alignment JSON"
            )
            continue
        if not slice_info.anchoring:
            continue

        coords_x = group["X"].values
        coords_y = group["Y"].values
        image_width = int(group["image_width"].iloc[0])
        image_height = int(group["image_height"].iloc[0])

        result = coordinates_to_atlas_space(
            pipeline_ctx,
            slice_info,
            coords_x,
            coords_y,
            image_width,
            image_height,
        )
        results.append(result)

    if not results:
        results = [SectionResult.empty()]

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

    return ExtractionResult(
        pixel_points=points,
        centroids=centroids,
        points_labels=points_labels,
        centroids_labels=centroids_labels,
        points_hemi_labels=points_hemi_labels,
        centroids_hemi_labels=centroids_hemi_labels,
        region_areas_list=region_areas_list,
        points_len=points_len,
        centroids_len=centroids_len,
        segmentation_filenames=[],
        per_point_undamaged=per_point_undamaged,
        per_centroid_undamaged=per_centroid_undamaged,
        total_points_len=total_points_len,
        total_centroids_len=total_centroids_len,
    )

"""Batch processing for folder-level atlas space transformation.

This module contains functions for processing all segmentation files
in a folder, mapping each one to atlas space using parallel execution.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from ...context import PipelineContext, SectionContext
from ...results import (
    SectionResult,
    IntensitySectionResult,
    ExtractionResult,
    PointSetResult,
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
from ..reorientation import reorient_points
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

                section_ctx = SectionContext(
                    section_number=seg_nr,
                    slice_info=slice_info,
                    segmentation_path=seg_path,
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


def _combine_region_areas(area_dfs):
    """Sum per-section region-area DataFrames into a single whole-series DF."""
    non_empty = [df for df in area_dfs if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    sum_cols = [c for c in numeric_cols if c != "idx"]
    return combined.groupby("idx")[sum_cols].sum().reset_index()


# Columns that are ratios — must be recomputed, not summed.
_RATIO_COLS = frozenset({
    "area_fraction", "left_hemi_area_fraction", "right_hemi_area_fraction",
    "undamaged_area_fraction", "left_hemi_undamaged_area_fraction",
    "right_hemi_undamaged_area_fraction",
    "mean_intensity", "left_hemi_mean_intensity", "right_hemi_mean_intensity",
})


def _combine_intensity_dfs(dfs):
    """Combine per-section intensity DataFrames into a single whole-series DF."""
    non_empty = [df for df in dfs if not df.empty and not df.dropna(how="all").empty]
    if not non_empty:
        return None
    group_cols = ["idx", "name", "r", "g", "b"]
    available_group_cols = [c for c in group_cols if c in non_empty[0].columns]

    combined = pd.concat(non_empty, ignore_index=True)
    for col in combined.columns:
        if col not in available_group_cols:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    sum_cols = [c for c in numeric_cols
                if c not in set(available_group_cols) and c not in _RATIO_COLS]

    return combined.groupby(available_group_cols)[sum_cols].sum().reset_index()


def _collect_section_results(results):
    """Reduce section results into concatenated arrays and combined region areas."""
    pts, ctrs = [], []
    pts_lbl, ctrs_lbl = [], []
    pts_hemi, ctrs_hemi = [], []
    pt_undam, ct_undam = [], []
    pts_len, ctrs_len = [], []
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
        areas.append(r.region_areas)

    return (
        _concat(pts, dtype=np.float64),
        _concat(ctrs, dtype=np.float64),
        _concat(pts_lbl, dtype=np.int64),
        _concat(ctrs_lbl, dtype=np.int64),
        _concat(pts_hemi, dtype=np.int64),
        _concat(ctrs_hemi, dtype=np.int64),
        _combine_region_areas(areas),
        pts_len,
        ctrs_len,
        _concat(pt_undam, dtype=bool),
        _concat(ct_undam, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Binary pipeline
# ---------------------------------------------------------------------------


def seg_to_coords(
    folder,
    registration: RegistrationData,
    atlas: AtlasData,
    pixel_id=[0, 0, 0],
    object_cutoff=0,
    non_linear=True,
    apply_damage_mask=True,
    segmentation_format="binary",
    return_orientation="asr",
):
    """Transform segmentation images into atlas-space coordinates.

    Parameters
    ----------
    folder
        Path to a folder containing segmentation images.
    registration
        Registration data returned by :func:`PyNutil.read_alignment`.
    atlas
        Atlas definition to use for labeling. This may be an
        :class:`~PyNutil.AtlasData` instance or a BrainGlobe atlas object.
    pixel_id
        RGB value or label identifier used to select the segmented class of
        interest.
    object_cutoff
        Minimum object size to keep during segmentation processing.
    non_linear
        If ``True``, apply non-linear deformation from the registration data
        when available.
    apply_damage_mask
        If ``True``, exclude damaged regions from filtered outputs and attach
        undamaged masks to the returned point sets.
    segmentation_format
        Name of the segmentation adapter to use, for example ``"binary"`` or
        ``"cellpose"``.
    return_orientation: 3-letter BrainGlobe orientation string (e.g. "asr",
            "ras"). Defaults to "asr" (internal orientation).

    Returns
    -------
    ExtractionResult
        Atlas-space points, centroid-level objects, section metadata, and
        region-area summaries for the processed series.
        The returned object exposes ``result.points`` for per-pixel
        atlas-space coordinates and ``result.objects`` for centroid-level
        object coordinates. Both point sets include labels, hemisphere labels,
        per-section lengths, and undamaged masks when available.

    Examples
    --------
    Process binary segmentation images with a BrainGlobe atlas:

    >>> from brainglobe_atlasapi import BrainGlobeAtlas
    >>> atlas = BrainGlobeAtlas("allen_mouse_25um")
    >>> registration = read_alignment("path/to/alignment.json")
    >>> result = seg_to_coords(
    ...     "path/to/segmentations/",
    ...     registration,
    ...     atlas,
    ...     pixel_id=[0, 0, 0],
    ... )
    >>> result.points.points.shape
    (N, 3)
    >>> result.objects.labels.shape
    (M,)
    """
    from ...io.atlas_loader import resolve_atlas
    atlas = resolve_atlas(atlas)
    atlas_shape = atlas.volume.shape
    pipeline_ctx = PipelineContext.from_format(
        segmentation_format=segmentation_format,
        atlas_labels=atlas.labels,
        atlas_volume=atlas.volume,
        hemi_map=atlas.hemi_map,
        non_linear=non_linear,
        object_cutoff=object_cutoff,
        pixel_id=pixel_id,
        apply_damage_mask=apply_damage_mask,
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
        region_areas,
        points_len,
        centroids_len,
        per_point_undamaged,
        per_centroid_undamaged,
    ) = _collect_section_results(results)

    if return_orientation != "lpi":
        #LPI is the internal orientation assumed by PyNutil
        #we keep this consistent as different orientations
        #can cause small rounding differences which effect
        #the results. keeping everything LPI makes Pynutil
        #reproducible.
        points = reorient_points(points, atlas_shape, return_orientation)
        centroids = reorient_points(centroids, atlas_shape, return_orientation)

    point_set = PointSetResult(
        points=points,
        labels=points_labels,
        hemi_labels=points_hemi_labels,
        section_lengths=points_len,
        undamaged_mask=per_point_undamaged,
        orientation=return_orientation,
        atlas_shape=atlas_shape,
    )
    object_set = PointSetResult(
        points=centroids,
        labels=centroids_labels,
        hemi_labels=centroids_hemi_labels,
        section_lengths=centroids_len,
        undamaged_mask=per_centroid_undamaged,
        orientation=return_orientation,
        atlas_shape=atlas_shape,
    )

    return ExtractionResult(
        points=point_set,
        objects=object_set,
        section_filenames=segmentations,
        region_areas=region_areas,
    )


def image_to_coords(
    folder,
    registration: RegistrationData,
    atlas: AtlasData,
    intensity_channel="grayscale",
    non_linear=True,
    apply_damage_mask=True,
    min_intensity=None,
    max_intensity=None,
    return_orientation="asr",
):
    """Transform image intensities into atlas-space point data.

    Parameters
    ----------
    folder
        Path to a folder containing source images.
    registration
        Registration data returned by :func:`PyNutil.read_alignment`.
    atlas
        Atlas definition to use for labeling. This may be an
        :class:`~PyNutil.AtlasData` instance or a BrainGlobe atlas object.
    intensity_channel
        Image channel to convert to intensity values, such as
        ``"grayscale"``.
    non_linear
        If ``True``, apply non-linear deformation from the registration data
        when available.
    apply_damage_mask
        If ``True``, exclude damaged regions from filtered outputs and attach
        undamaged masks to the returned point sets.
    min_intensity
        Optional lower threshold. Intensities below this value are discarded.
    max_intensity
        Optional upper threshold. Intensities above this value are discarded.
    return_orientation: 3-letter BrainGlobe orientation string (e.g. "asr",
            "ras"). Defaults to "asr" (internal orientation).

    Returns
    -------
    ExtractionResult
        Atlas-space point data with optional per-point intensity values and
        aggregated per-region intensity summaries.
        The atlas-space coordinates are stored in ``result.points.points`` and
        the sampled intensities in ``result.points.point_values``.
        Per-region intensity summaries, when present, are stored in
        ``result.region_intensities``.

    Examples
    --------
    Quantify image intensity instead of segmented objects:

    >>> from brainglobe_atlasapi import BrainGlobeAtlas
    >>> atlas = BrainGlobeAtlas("allen_mouse_25um")
    >>> registration = read_alignment("path/to/alignment.json")
    >>> result = image_to_coords(
    ...     "path/to/images/",
    ...     registration,
    ...     atlas,
    ... )
    >>> result.points.points.shape
    (N, 3)
    >>> result.region_intensities.columns.tolist()[:3]
    ['idx', 'name', 'r']
    """
    from ...io.atlas_loader import resolve_atlas
    atlas = resolve_atlas(atlas)
    atlas_shape = atlas.volume.shape
    pipeline_ctx = PipelineContext.from_format(
        segmentation_format="binary",
        atlas_labels=atlas.labels,
        atlas_volume=atlas.volume,
        hemi_map=atlas.hemi_map,
        non_linear=non_linear,
        object_cutoff=0,
        pixel_id=[0, 0, 0],
        apply_damage_mask=apply_damage_mask,
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
    region_intensities_list = [r.region_intensities for r in results
                               if r.region_intensities is not None]

    all_points = _concat([r.points for r in results], none_if_empty=True)
    all_labels = _concat([r.points_labels for r in results], none_if_empty=True)
    all_hemi = _concat([r.points_hemi_labels for r in results], none_if_empty=True)
    all_intensities = _concat(
        [r.point_intensities for r in results], none_if_empty=True
    )
    points_len = [r.num_points for r in results]

    if return_orientation != "lpi":
        all_points = reorient_points(all_points, atlas_shape, return_orientation)

    point_set = PointSetResult(
        points=all_points,
        labels=all_labels,
        hemi_labels=all_hemi,
        section_lengths=points_len,
        point_values=all_intensities,
        orientation=return_orientation,
        atlas_shape=atlas_shape,
    )

    # Combine per-section intensity DataFrames into a single whole-series DF.
    combined_intensities = None
    if region_intensities_list:
        combined_intensities = _combine_intensity_dfs(region_intensities_list)

    return ExtractionResult(
        points=point_set,
        objects=None,
        section_filenames=images,
        region_intensities=combined_intensities,
    )


# ---------------------------------------------------------------------------
# Coordinate pipeline
# ---------------------------------------------------------------------------


def xy_to_coords(
    coordinate_file,
    registration: RegistrationData,
    atlas: AtlasData,
    non_linear=True,
    apply_damage_mask=True,
    return_orientation="asr",
):
    """Transform image-space coordinates from CSV into atlas space.

    Parameters
    ----------
    coordinate_file
        Path to a CSV file containing coordinates and section metadata. The
        file is expected to contain the columns ``X``, ``Y``,
        ``image_width``, ``image_height``, and ``section number``.
    registration
        Registration data returned by :func:`PyNutil.read_alignment`.
    atlas
        Atlas definition to use for labeling. This may be an
        :class:`~PyNutil.AtlasData` instance or a BrainGlobe atlas object.
    non_linear
        If ``True``, apply non-linear deformation from the registration data
        when available.
    apply_damage_mask
        If ``True``, exclude damaged regions from filtered outputs and attach
        undamaged masks to the returned point sets.
    return_orientation: 3-letter BrainGlobe orientation string (e.g. "asr",
            "ras"). Defaults to "asr" (internal orientation).

    Returns
    -------
    ExtractionResult
        Atlas-space points, object placeholders, and region-area summaries
        derived from the input coordinates.
        In coordinate mode, ``result.points`` contains the transformed
        atlas-space coordinates and labels, while ``result.objects`` mirrors
        the same coordinates for downstream quantification and export code.

    Examples
    --------
    Transform pre-extracted image-space coordinates from CSV:

    >>> from brainglobe_atlasapi import BrainGlobeAtlas
    >>> atlas = BrainGlobeAtlas("allen_mouse_25um")
    >>> registration = read_alignment("path/to/alignment.json")
    >>> result = xy_to_coords(
    ...     "path/to/coordinates.csv",
    ...     registration,
    ...     atlas,
    ... )
    >>> result.points.points.shape
    (N, 3)
    >>> result.section_filenames
    []
    """
    from ...io.atlas_loader import resolve_atlas
    atlas = resolve_atlas(atlas)
    atlas_shape = atlas.volume.shape
    from ...io.loaders import load_coordinate_file

    coord_df = load_coordinate_file(coordinate_file)

    slices_by_nr = {s.section_number: s for s in registration.slices}

    # Build a minimal PipelineContext (no segmentation adapter needed for coordinates)
    pipeline_ctx = PipelineContext.from_format(
        segmentation_format="binary",
        atlas_labels=atlas.labels,
        atlas_volume=atlas.volume,
        hemi_map=atlas.hemi_map,
        non_linear=non_linear,
        object_cutoff=0,
        pixel_id=[0, 0, 0],
        apply_damage_mask=apply_damage_mask,
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
        region_areas,
        points_len,
        centroids_len,
        per_point_undamaged,
        per_centroid_undamaged,
    ) = _collect_section_results(results)

    if return_orientation != "lpi":
        points = reorient_points(points, atlas_shape, return_orientation)
        centroids = reorient_points(centroids, atlas_shape, return_orientation)

    point_set = PointSetResult(
        points=points,
        labels=points_labels,
        hemi_labels=points_hemi_labels,
        section_lengths=points_len,
        undamaged_mask=per_point_undamaged,
        orientation=return_orientation,
        atlas_shape=atlas_shape,
    )
    object_set = PointSetResult(
        points=centroids,
        labels=centroids_labels,
        hemi_labels=centroids_hemi_labels,
        section_lengths=centroids_len,
        undamaged_mask=per_centroid_undamaged,
        orientation=return_orientation,
        atlas_shape=atlas_shape,
    )

    return ExtractionResult(
        points=point_set,
        objects=object_set,
        section_filenames=[],
        region_areas=region_areas,
    )

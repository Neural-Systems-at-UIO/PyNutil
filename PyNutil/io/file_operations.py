import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from .meshview_writer import write_hemi_points_to_meshview
from .atlas_loader import resolve_atlas_labels


@dataclass
class SaveContext:
    """Groups the parameters needed by :func:`save_analysis_output`."""

    # Core data
    points: Optional[np.ndarray] = None
    objects: Optional[np.ndarray] = None
    label_df: Optional[pd.DataFrame] = None
    per_section_df: Optional[list] = None
    point_labels: Optional[np.ndarray] = None
    object_labels: Optional[np.ndarray] = None
    points_hemi_labels: Optional[np.ndarray] = None
    objects_hemi_labels: Optional[np.ndarray] = None
    points_len: Optional[list] = None
    objects_len: Optional[list] = None
    section_filenames: Optional[list] = None
    atlas_labels: Optional[pd.DataFrame] = None
    point_values: Optional[np.ndarray] = None

    # Whether this is intensity mode (affects report filename)
    is_intensity: bool = False

    # Settings dict written to pynutil_settings.json (optional)
    settings_dict: Optional[Dict[str, Any]] = None

    # Output control
    prepend: str = ""
    colormap: str = "gray"


@dataclass
class _SectionWindow:
    """Per-section slicing window for report and meshview outputs."""

    stem: str
    points_start: int
    points_end: int
    centroids_start: int
    centroids_end: int
    df: pd.DataFrame


def _slice_or_none(arr, start: int, end: int):
    """Slice *arr* if present, otherwise return None."""
    return arr[start:end] if arr is not None else None


def _iter_section_windows(ctx: SaveContext):
    """Yield section-aligned slicing windows for per-section outputs."""
    if ctx.section_filenames is None or ctx.per_section_df is None:
        return

    points_len = ctx.points_len or [0] * len(ctx.section_filenames)
    objects_len = ctx.objects_len or [0] * len(ctx.section_filenames)

    points_offset = 0
    objects_offset = 0
    for pl, ol, fn, df in zip(
        points_len,
        objects_len,
        ctx.section_filenames,
        ctx.per_section_df,
    ):
        stem = os.path.basename(fn).split(".")[0]
        yield _SectionWindow(
            stem=stem,
            points_start=points_offset,
            points_end=points_offset + pl,
            centroids_start=objects_offset,
            centroids_end=objects_offset + ol,
            df=df,
        )
        points_offset += pl
        objects_offset += ol


def _ensure_analysis_output_dirs(output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for subdir in (
        "whole_series_report",
        "per_section_meshview",
        "per_section_reports",
        "whole_series_meshview",
    ):
        os.makedirs(f"{output_folder}/{subdir}", exist_ok=True)


def save_analysis_output(ctx: SaveContext, output_folder: str):
    """
    Save the analysis output to the specified folder.

    Parameters
    ----------
    ctx : SaveContext
        All data and configuration needed for saving.
    output_folder : str
        The folder where the output will be saved.
    """
    # Create the output folder if it doesn't exist
    _ensure_analysis_output_dirs(output_folder)

    if ctx.label_df is not None:
        report_name = "intensity.csv" if ctx.is_intensity else "counts.csv"
        ctx.label_df.to_csv(
            f"{output_folder}/whole_series_report/{ctx.prepend}{report_name}",
            sep=";",
            na_rep="",
            index=False,
        )
    elif not ctx.prepend:
        print("No quantification found, so only coordinates will be saved.")
        print(
            "If you want to save the quantification, please run quantify_coordinates."
        )

    if ctx.per_section_df is not None and ctx.section_filenames is not None:
        _save_per_section_reports(ctx, output_folder)
    if ctx.points is not None:
        _save_whole_series_meshview(ctx, output_folder)

    _save_settings_json(ctx, output_folder)


def _save_settings_json(ctx: SaveContext, output_folder: str) -> None:
    """Write a reference settings JSON to *output_folder*."""
    if ctx.settings_dict is None:
        return
    settings_file_path = os.path.join(output_folder, "pynutil_settings.json")
    with open(settings_file_path, "w") as f:
        json.dump(ctx.settings_dict, f, indent=4)


def _save_per_section_reports(ctx: SaveContext, output_folder: str):
    """Write per-section CSVs and MeshView JSONs."""
    for window in _iter_section_windows(ctx):
        window.df.to_csv(
            f"{output_folder}/per_section_reports/{ctx.prepend}{window.stem}.csv",
            sep=";",
            na_rep="",
            index=False,
        )
        if ctx.points is not None or ctx.objects is not None:
            section_values = (
                _slice_or_none(ctx.point_values, window.points_start, window.points_end)
                if ctx.point_values is not None
                else None
            )
            write_hemi_points_to_meshview(
                _slice_or_none(ctx.points, window.points_start, window.points_end),
                _slice_or_none(ctx.point_labels, window.points_start, window.points_end),
                _slice_or_none(ctx.points_hemi_labels, window.points_start, window.points_end),
                f"{output_folder}/per_section_meshview/{ctx.prepend}{window.stem}_pixels.json",
                ctx.atlas_labels,
                section_values,
                colormap=ctx.colormap,
            )
            if ctx.objects is not None:
                write_hemi_points_to_meshview(
                    _slice_or_none(ctx.objects, window.centroids_start, window.centroids_end),
                    _slice_or_none(
                        ctx.object_labels,
                        window.centroids_start,
                        window.centroids_end,
                    ),
                    _slice_or_none(
                        ctx.objects_hemi_labels,
                        window.centroids_start,
                        window.centroids_end,
                    ),
                    f"{output_folder}/per_section_meshview/{ctx.prepend}{window.stem}_centroids.json",
                    ctx.atlas_labels,
                    colormap=ctx.colormap,
                )


def _save_whole_series_meshview(ctx: SaveContext, output_folder: str):
    """Write whole-series MeshView JSONs for pixels and centroids."""
    write_hemi_points_to_meshview(
        ctx.points,
        ctx.point_labels,
        ctx.points_hemi_labels,
        f"{output_folder}/whole_series_meshview/{ctx.prepend}pixels_meshview.json",
        ctx.atlas_labels,
        ctx.point_values,
        colormap=ctx.colormap,
    )
    if ctx.objects is not None:
        write_hemi_points_to_meshview(
            ctx.objects,
            ctx.object_labels,
            ctx.objects_hemi_labels,
            f"{output_folder}/whole_series_meshview/{ctx.prepend}objects_meshview.json",
            ctx.atlas_labels,
            colormap=ctx.colormap,
        )


def save_analysis(
    output_folder,
    result,
    atlas_labels,
    label_df=None,
    per_section_df=None,
    *,
    colormap="gray",
    settings_dict=None,
):
    """Save analysis output to the specified directory.

    Args:
        output_folder: Directory to write output files.
        result: ExtractionResult from coordinate extraction.
        atlas_labels: Atlas labels DataFrame (or AtlasData — ``.labels`` used).
        label_df: Whole-series quantification DataFrame.
        per_section_df: List of per-section DataFrames.
        colormap: Colormap for MeshView intensity output.
        settings_dict: Optional dict written to pynutil_settings.json.
    """
    atlas_labels = resolve_atlas_labels(atlas_labels)

    ctx = SaveContext(
        points=result.points.filtered_points() if result else None,
        objects=(
            result.objects.filtered_points()
            if (result and result.objects is not None)
            else None
        ),
        label_df=label_df,
        per_section_df=per_section_df,
        point_labels=result.points.filtered_labels() if result else None,
        object_labels=(
            result.objects.filtered_labels()
            if (result and result.objects is not None)
            else None
        ),
        points_hemi_labels=result.points.filtered_hemi_labels() if result else None,
        objects_hemi_labels=(
            result.objects.filtered_hemi_labels()
            if (result and result.objects is not None)
            else None
        ),
        points_len=result.points.filtered_section_lengths() if result else None,
        objects_len=(
            result.objects.filtered_section_lengths()
            if (result and result.objects is not None)
            else None
        ),
        section_filenames=result.section_filenames if result else None,
        atlas_labels=atlas_labels,
        point_values=result.points.filtered_point_values() if result else None,
        is_intensity=result.region_intensities is not None if result else False,
        settings_dict=settings_dict,
        colormap=colormap,
    )
    save_analysis_output(ctx, output_folder)

    # Remap compressed IDs if present
    if label_df is not None and "original_idx" in label_df.columns:
        remapped = label_df.copy()
        remapped["idx"] = remapped["original_idx"]
        remapped = remapped.drop(columns=["original_idx"])
        remapped.to_csv(
            f"{output_folder}/whole_series_report/counts.csv",
            sep=";",
            index=False,
        )

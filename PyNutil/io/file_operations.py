import os
import json
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

from .meshview_writer import write_hemi_points_to_meshview


@dataclass
class SaveContext:
    """Groups the many parameters needed by :func:`save_analysis_output`.

    Replaces 26+ positional parameters with a single, documented object.
    """

    # Core data
    pixel_points: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None
    label_df: Optional[pd.DataFrame] = None
    per_section_df: Optional[list] = None
    labeled_points: Optional[np.ndarray] = None
    labeled_points_centroids: Optional[np.ndarray] = None
    points_hemi_labels: Optional[np.ndarray] = None
    centroids_hemi_labels: Optional[np.ndarray] = None
    points_len: Optional[list] = None
    centroids_len: Optional[list] = None
    segmentation_filenames: Optional[list] = None
    atlas_labels: Optional[pd.DataFrame] = None
    point_intensities: Optional[np.ndarray] = None

    # Configuration / metadata (saved to settings JSON)
    segmentation_folder: Optional[str] = None
    image_folder: Optional[str] = None
    alignment_json: Optional[str] = None
    colour: Optional[list] = None
    intensity_channel: Optional[str] = None
    atlas_name: Optional[str] = None
    custom_region_path: Optional[str] = None
    atlas_path: Optional[str] = None
    label_path: Optional[str] = None
    settings_file: Optional[str] = None

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
    if ctx.segmentation_filenames is None or ctx.per_section_df is None:
        return

    points_len = ctx.points_len or [0] * len(ctx.segmentation_filenames)
    centroids_len = ctx.centroids_len or [0] * len(ctx.segmentation_filenames)

    points_offset = 0
    centroids_offset = 0
    for pl, cl, fn, df in zip(
        points_len,
        centroids_len,
        ctx.segmentation_filenames,
        ctx.per_section_df,
    ):
        stem = os.path.basename(fn).split(".")[0]
        yield _SectionWindow(
            stem=stem,
            points_start=points_offset,
            points_end=points_offset + pl,
            centroids_start=centroids_offset,
            centroids_end=centroids_offset + cl,
            df=df,
        )
        points_offset += pl
        centroids_offset += cl


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
        report_name = "intensity.csv" if ctx.image_folder else "counts.csv"
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

    if ctx.per_section_df is not None and ctx.segmentation_filenames is not None:
        _save_per_section_reports(ctx, output_folder)
    if ctx.pixel_points is not None:
        _save_whole_series_meshview(ctx, output_folder)

    _save_settings_json(ctx, output_folder)


def _save_settings_json(ctx: SaveContext, output_folder: str) -> None:
    """Write a reference settings JSON to *output_folder*."""
    settings_dict = {
        "segmentation_folder": ctx.segmentation_folder,
        "image_folder": ctx.image_folder,
        "alignment_json": ctx.alignment_json,
        "colour": ctx.colour,
        "intensity_channel": ctx.intensity_channel,
        "custom_region_path": ctx.custom_region_path,
    }

    for key, val in [
        ("atlas_name", ctx.atlas_name),
        ("atlas_path", ctx.atlas_path),
        ("label_path", ctx.label_path),
        ("settings_file", ctx.settings_file),
    ]:
        if val:
            settings_dict[key] = val

    settings_file_path = os.path.join(output_folder, "pynutil_settings.json")
    with open(settings_file_path, "w") as f:
        json.dump(settings_dict, f, indent=4)


def _save_per_section_reports(ctx: SaveContext, output_folder: str):
    """Write per-section CSVs and MeshView JSONs."""
    for window in _iter_section_windows(ctx):
        window.df.to_csv(
            f"{output_folder}/per_section_reports/{ctx.prepend}{window.stem}.csv",
            sep=";",
            na_rep="",
            index=False,
        )
        if ctx.pixel_points is not None or ctx.centroids is not None:
            section_intensities = (
                _slice_or_none(ctx.point_intensities, window.points_start, window.points_end)
                if ctx.point_intensities is not None
                else None
            )
            _save_per_section_meshview(
                ctx,
                output_folder,
                window,
                section_intensities,
            )


def _save_per_section_meshview(
    ctx: SaveContext,
    output_folder: str,
    window: _SectionWindow,
    section_intensities=None,
):
    """Write per-section MeshView JSONs for pixels and centroids."""
    write_hemi_points_to_meshview(
        _slice_or_none(ctx.pixel_points, window.points_start, window.points_end),
        _slice_or_none(ctx.labeled_points, window.points_start, window.points_end),
        _slice_or_none(
            ctx.points_hemi_labels,
            window.points_start,
            window.points_end,
        ),
        f"{output_folder}/per_section_meshview/{ctx.prepend}{window.stem}_pixels.json",
        ctx.atlas_labels,
        section_intensities,
        colormap=ctx.colormap,
    )
    if ctx.centroids is not None:
        write_hemi_points_to_meshview(
            _slice_or_none(
                ctx.centroids,
                window.centroids_start,
                window.centroids_end,
            ),
            _slice_or_none(
                ctx.labeled_points_centroids,
                window.centroids_start,
                window.centroids_end,
            ),
            _slice_or_none(
                ctx.centroids_hemi_labels,
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
        ctx.pixel_points,
        ctx.labeled_points,
        ctx.points_hemi_labels,
        f"{output_folder}/whole_series_meshview/{ctx.prepend}pixels_meshview.json",
        ctx.atlas_labels,
        ctx.point_intensities,
        colormap=ctx.colormap,
    )
    if ctx.centroids is not None:
        write_hemi_points_to_meshview(
            ctx.centroids,
            ctx.labeled_points_centroids,
            ctx.centroids_hemi_labels,
            f"{output_folder}/whole_series_meshview/{ctx.prepend}objects_meshview.json",
            ctx.atlas_labels,
            colormap=ctx.colormap,
        )

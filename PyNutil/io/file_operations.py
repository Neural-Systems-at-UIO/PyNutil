import os
import json
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .meshview_writer import write_hemi_points_to_meshview

if TYPE_CHECKING:
    from ..config import PyNutilConfig


@dataclass
class SaveContext:
    """Groups the parameters needed by :func:`save_analysis_output`.

    Configuration metadata (paths, colour, etc.) is passed as a single
    ``config`` reference rather than as individual fields.
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

    # Single config reference (replaces ~10 individual fields)
    config: Optional["PyNutilConfig"] = None

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
        cfg = ctx.config
        report_name = "intensity.csv" if (cfg and cfg.image_folder) else "counts.csv"
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
    cfg = ctx.config
    if cfg is None:
        return
    settings_dict = {
        "segmentation_folder": cfg.segmentation_folder,
        "image_folder": cfg.image_folder,
        "alignment_json": cfg.alignment_json,
        "colour": cfg.colour,
        "intensity_channel": cfg.intensity_channel,
        "custom_region_path": cfg.custom_region_path,
    }

    for key, val in [
        ("atlas_name", cfg.atlas_name),
        ("atlas_path", cfg.atlas_path),
        ("label_path", cfg.label_path),
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
            write_hemi_points_to_meshview(
                _slice_or_none(ctx.pixel_points, window.points_start, window.points_end),
                _slice_or_none(ctx.labeled_points, window.points_start, window.points_end),
                _slice_or_none(ctx.points_hemi_labels, window.points_start, window.points_end),
                f"{output_folder}/per_section_meshview/{ctx.prepend}{window.stem}_pixels.json",
                ctx.atlas_labels,
                section_intensities,
                colormap=ctx.colormap,
            )
            if ctx.centroids is not None:
                write_hemi_points_to_meshview(
                    _slice_or_none(ctx.centroids, window.centroids_start, window.centroids_end),
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

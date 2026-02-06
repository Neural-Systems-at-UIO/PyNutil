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
    prev_pl = 0
    prev_cl = 0

    # Handle None for points_len and centroids_len (e.g. in intensity mode)
    points_len = ctx.points_len or [0] * len(ctx.segmentation_filenames)
    centroids_len = ctx.centroids_len or [0] * len(ctx.segmentation_filenames)

    for pl, cl, fn, df in zip(
        points_len,
        centroids_len,
        ctx.segmentation_filenames,
        ctx.per_section_df,
    ):
        split_fn = fn.split(os.sep)[-1].split(".")[0]
        df.to_csv(
            f"{output_folder}/per_section_reports/{ctx.prepend}{split_fn}.csv",
            sep=";",
            na_rep="",
            index=False,
        )
        if ctx.pixel_points is not None or ctx.centroids is not None:
            section_intensities = (
                ctx.point_intensities[prev_pl : pl + prev_pl]
                if ctx.point_intensities is not None
                else None
            )
            _save_per_section_meshview(
                ctx, output_folder, split_fn, pl, cl, prev_pl, prev_cl,
                section_intensities,
            )
        prev_cl += cl
        prev_pl += pl


def _save_per_section_meshview(
    ctx: SaveContext,
    output_folder: str,
    split_fn: str,
    pl: int,
    cl: int,
    prev_pl: int,
    prev_cl: int,
    section_intensities=None,
):
    """Write per-section MeshView JSONs for pixels and centroids."""
    write_hemi_points_to_meshview(
        ctx.pixel_points[prev_pl : pl + prev_pl] if ctx.pixel_points is not None else None,
        ctx.labeled_points[prev_pl : pl + prev_pl] if ctx.labeled_points is not None else None,
        ctx.points_hemi_labels[prev_pl : pl + prev_pl]
        if ctx.points_hemi_labels is not None
        else None,
        f"{output_folder}/per_section_meshview/{ctx.prepend}{split_fn}_pixels.json",
        ctx.atlas_labels,
        section_intensities,
        colormap=ctx.colormap,
    )
    if ctx.centroids is not None:
        write_hemi_points_to_meshview(
            ctx.centroids[prev_cl : cl + prev_cl],
            ctx.labeled_points_centroids[prev_cl : cl + prev_cl],
            ctx.centroids_hemi_labels[prev_cl : cl + prev_cl],
            f"{output_folder}/per_section_meshview/{ctx.prepend}{split_fn}_centroids.json",
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

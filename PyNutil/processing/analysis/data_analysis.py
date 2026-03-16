"""Quantification and aggregation of atlas-space labelled data.

Public API
----------
- :func:`quantify_labeled_points` — aggregate per-pixel/centroid counts.
- :func:`quantify_intensity` — aggregate per-region intensity.
- :func:`map_to_custom_regions` — remap point labels to custom regions.
- :func:`apply_custom_regions` — remap a region DataFrame to custom regions.
"""

import numpy as np
import pandas as pd

from ...results import PerEntityArrays
from .region_counting import pixel_count_per_region
from ..utils import (
    AREA_FRACTION_PAIRS,
    apply_area_fractions,
    apply_mean_intensities,
    reindex_to_atlas,
)

# Columns that are ratios and must be recomputed after summing, not summed themselves.
_RATIO_COLS = frozenset(
    {
        "area_fraction",
        "left_hemi_area_fraction",
        "right_hemi_area_fraction",
        "undamaged_area_fraction",
        "left_hemi_undamaged_area_fraction",
        "right_hemi_undamaged_area_fraction",
        "mean_intensity",
        "left_hemi_mean_intensity",
        "right_hemi_mean_intensity",
    }
)


# ── Custom region helpers ────────────────────────────────────────────────



def map_to_custom_regions(custom_regions_dict, points_labels):
    """Reassign atlas-region labels to user-defined custom region IDs.

    Args:
        custom_regions_dict: Dict with keys ``custom_ids``, ``custom_names``,
            ``rgb_values``, ``subregion_ids``.
        points_labels: 1-D array of atlas region IDs (one per point).

    Returns:
        1-D array of custom region IDs (same shape as *points_labels*).
    """
    custom_points_labels = np.zeros_like(points_labels)
    for i in np.unique(points_labels):
        new_id = np.where([i in r for r in custom_regions_dict["subregion_ids"]])[0]
        if len(new_id) > 1:
            raise ValueError(f"error, region id {i} is in more than one custom region")
        if len(new_id) == 0:
            continue
        new_id = custom_regions_dict["custom_ids"][new_id[0]]
        custom_points_labels[points_labels == i] = int(new_id)
    return custom_points_labels


def apply_custom_regions(df, custom_regions_dict):
    """Remap a region-level DataFrame to user-defined custom regions.

    Args:
        df: DataFrame with at least an ``idx`` column containing atlas
            region IDs, plus various count/area columns.
        custom_regions_dict: Dict with keys ``custom_ids``, ``custom_names``,
            ``rgb_values``, ``subregion_ids``.

    Returns:
        (grouped_df, df) — *grouped_df* aggregated by custom region,
        *df* with an added ``custom_region_name`` column.
    """
    id_mapping = {}
    name_mapping = {}
    rgb_mapping = {}
    for cid, cname, rgb, subregions in zip(
        custom_regions_dict["custom_ids"],
        custom_regions_dict["custom_names"],
        custom_regions_dict["rgb_values"],
        custom_regions_dict["subregion_ids"],
    ):
        for sid in subregions:
            id_mapping[sid] = cid
            name_mapping[sid] = cname
            rgb_mapping[sid] = rgb

    # Annotate original df
    df["custom_region_name"] = df["idx"].map(name_mapping).fillna("")
    temp_df = df.copy()
    temp_df["r"] = temp_df["idx"].map(lambda x: rgb_mapping[x][0] if x in rgb_mapping else None)
    temp_df["g"] = temp_df["idx"].map(lambda x: rgb_mapping[x][1] if x in rgb_mapping else None)
    temp_df["b"] = temp_df["idx"].map(lambda x: rgb_mapping[x][2] if x in rgb_mapping else None)
    temp_df["idx"] = temp_df["idx"].map(id_mapping)

    # Aggregate all numeric columns dynamically
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in {"idx", "r", "g", "b"}]
    agg_dict = {col: "sum" for col in numeric_cols}
    agg_dict.update({"r": "first", "g": "first", "b": "first"})

    grouped_df = (
        temp_df[temp_df["custom_region_name"] != ""]
        .groupby("custom_region_name", dropna=True)
        .agg(agg_dict)
        .reset_index()
    )
    grouped_df = grouped_df.rename(columns={"custom_region_name": "name"})

    # Compute all area-fraction and mean-intensity derived columns
    apply_area_fractions(grouped_df)
    apply_mean_intensities(grouped_df)

    common_columns = [col for col in df.columns if col in grouped_df.columns]
    grouped_df = grouped_df.reindex(
        columns=common_columns
        + [col for col in grouped_df.columns if col not in common_columns]
    )
    return grouped_df, df


# ── Segmentation quantification ─────────────────────────────────────────


def quantify_labeled_points(
    points: PerEntityArrays,
    centroids: PerEntityArrays,
    region_areas_list,
    atlas_labels,
    apply_damage_mask,
):
    """Aggregate per-pixel and per-centroid counts into summary tables.

    Args:
        points: Concatenated per-pixel arrays (labels, hemi, undamaged, lengths).
        centroids: Concatenated per-centroid arrays (same structure).
        region_areas_list: List of region-area DataFrames per section.
        atlas_labels: Atlas labels DataFrame.
        apply_damage_mask: Whether damage mask was applied.

    Returns:
        (label_df, per_section_df) — whole-series and per-section DataFrames.
    """
    per_section_df = _quantify_per_section(
        points,
        centroids,
        region_areas_list,
        atlas_labels,
        apply_damage_mask,
    )

    def _area_fractions(df):
        for num, den, res in AREA_FRACTION_PAIRS:
            if num in df.columns and den in df.columns:
                df[res] = df[num] / df[den]

    label_df = _combine_reports(per_section_df, atlas_labels, derive_fn=_area_fractions)
    if not apply_damage_mask:
        cols = [c for c in label_df.columns if "damage" not in c]
        label_df = label_df[cols]
        per_section_df = [s[cols] for s in per_section_df]
    return label_df, per_section_df


def _quantify_per_section(
    points: PerEntityArrays,
    centroids: PerEntityArrays,
    region_areas_list,
    atlas_labels,
    with_damage=False,
):
    """Quantify counts per section, merging with region areas."""
    per_section_df = []

    for (p_lab, p_hemi, p_und), (c_lab, c_hemi, c_und), ra in zip(
        points.split(), centroids.split(), region_areas_list
    ):
        current_df = pixel_count_per_region(
            p_lab,
            c_lab,
            p_und,
            c_und,
            p_hemi,
            c_hemi,
            atlas_labels,
            with_damage,
        )
        current_df_new = _merge_dataframes(current_df, ra, atlas_labels)
        per_section_df.append(current_df_new)

    return per_section_df


def _merge_dataframes(current_df, ra, atlas_labels):
    """Merge count DataFrame with region areas and atlas labels."""
    cols_to_use = ra.columns.difference(atlas_labels.columns)
    all_region_df = atlas_labels.merge(ra[["idx", *cols_to_use]], on="idx", how="left")
    cols_to_use = current_df.columns.difference(all_region_df.columns)
    result = all_region_df.merge(
        current_df[["idx", *cols_to_use]], on="idx", how="left"
    )
    # Append rows from current_df not in atlas_labels (e.g. out_of_atlas).
    extra_idx = set(current_df["idx"]) - set(all_region_df["idx"])
    if extra_idx:
        extra = current_df[current_df["idx"].isin(extra_idx)].copy()
        for col in result.columns:
            if col not in extra.columns:
                extra[col] = 0
        result = pd.concat([result, extra[result.columns]], ignore_index=True)

    # Fill count/area numeric columns with 0 using shared area-fraction specs.
    base_numeric_cols = sorted({c for num, den, _ in AREA_FRACTION_PAIRS for c in (num, den)})
    for col in base_numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col]).fillna(0)

    apply_area_fractions(result)
    return result


# ── Intensity quantification ────────────────────────────────────────────


def quantify_intensity(region_intensities_list, atlas_labels):
    """Aggregate per-region intensity across sections.

    Args:
        region_intensities_list: List of per-section intensity DataFrames
            (may contain ``None`` for skipped sections).
        atlas_labels: Atlas labels DataFrame.

    Returns:
        (label_df, per_section_df) — whole-series and per-section DataFrames.
    """
    region_intensities_list = [df for df in region_intensities_list if df is not None]
    if not region_intensities_list:
        return pd.DataFrame(), []

    label_df = _combine_reports(
        region_intensities_list, atlas_labels, derive_fn=apply_mean_intensities
    )
    return label_df, region_intensities_list


# ── Shared combine logic ────────────────────────────────────────────────


def _combine_reports(per_section_df, atlas_labels, *, derive_fn):
    """Combine per-section DataFrames into a whole-series report.

    Shared by both the segmentation (area-fraction) and intensity
    (mean-intensity) pipelines.

    Args:
        per_section_df: List of per-section DataFrames.
        atlas_labels: Atlas labels DataFrame.
        derive_fn: Callable ``(df) → None`` that adds derived columns in-place.

    Returns:
        Combined DataFrame reindexed to all atlas regions.
    """
    group_cols = ["idx", "name", "r", "g", "b"]
    available_group_cols = [c for c in group_cols if c in per_section_df[0].columns]

    non_empty = [
        df for df in per_section_df
        if df is not None and not df.empty and not df.dropna(how="all").empty
    ]
    combined = pd.concat(non_empty) if non_empty else pd.DataFrame()
    for col in combined.columns:
        if col not in available_group_cols:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    sum_cols = [c for c in numeric_cols if c not in set(available_group_cols)]
    sum_cols = [c for c in sum_cols if c not in _RATIO_COLS]

    label_df = combined.groupby(available_group_cols)[sum_cols].sum().reset_index()

    derive_fn(label_df)
    label_df.fillna(0, inplace=True)
    label_df = reindex_to_atlas(label_df, atlas_labels)

    # Fill missing name/colour columns from atlas_labels for zero-count regions
    if "name" in atlas_labels.columns and "name" in label_df.columns:
        idx_map = atlas_labels.set_index("idx")
        for col in ["name", "r", "g", "b"]:
            if col in atlas_labels.columns:
                label_df[col] = label_df[col].fillna(label_df["idx"].map(idx_map[col]))

    label_df.fillna(0, inplace=True)
    return label_df


# ── Unified quantification entry point ──────────────────────────────────


def quantify_coords(result, atlas_labels, apply_damage_mask=True):
    """Quantify an ExtractionResult by atlas region.

    Dispatches to :func:`quantify_labeled_points` or :func:`quantify_intensity`
    depending on the content of *result*.

    Args:
        result: ExtractionResult from a coordinate extraction function.
        atlas_labels: Atlas labels DataFrame (or AtlasData — ``.labels``
            will be used).
        apply_damage_mask: Include damage statistics in output.

    Returns:
        ``(label_df, per_section_df)`` — whole-series and per-section DataFrames.
    """
    # Accept AtlasData or a raw DataFrame
    if hasattr(atlas_labels, "labels"):
        atlas_labels = atlas_labels.labels

    if result.region_intensities_list is not None:
        return quantify_intensity(result.region_intensities_list, atlas_labels)

    return quantify_labeled_points(
        PerEntityArrays(
            labels=result.points_labels,
            hemi_labels=result.points_hemi_labels,
            undamaged=result.per_point_undamaged,
            section_lengths=result.total_points_len,
        ),
        PerEntityArrays(
            labels=result.centroids_labels,
            hemi_labels=result.centroids_hemi_labels,
            undamaged=result.per_centroid_undamaged,
            section_lengths=result.total_centroids_len,
        ),
        result.region_areas_list,
        atlas_labels,
        apply_damage_mask,
    )

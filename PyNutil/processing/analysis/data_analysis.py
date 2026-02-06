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


def _build_custom_region_mappings(custom_regions_dict):
    """Build id/name/rgb lookup dicts from a custom regions definition.

    Args:
        custom_regions_dict: Dict with keys ``custom_ids``, ``custom_names``,
            ``rgb_values``, ``subregion_ids``.

    Returns:
        (id_mapping, name_mapping, rgb_mapping) — each maps *subregion id* →
        custom value.
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
    return id_mapping, name_mapping, rgb_mapping


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
    id_mapping, name_mapping, rgb_mapping = _build_custom_region_mappings(
        custom_regions_dict
    )

    # Annotate original df
    df["custom_region_name"] = df["idx"].map(name_mapping).fillna("")
    temp_df = df.copy()
    _apply_rgb_mapping(temp_df, id_mapping, rgb_mapping)

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


def _apply_rgb_mapping(temp_df, id_mapping, rgb_mapping):
    """Set r/g/b columns and remap idx using *id_mapping* and *rgb_mapping*."""
    temp_df["r"] = temp_df["idx"].map(
        lambda x: rgb_mapping[x][0] if x in rgb_mapping else None
    )
    temp_df["g"] = temp_df["idx"].map(
        lambda x: rgb_mapping[x][1] if x in rgb_mapping else None
    )
    temp_df["b"] = temp_df["idx"].map(
        lambda x: rgb_mapping[x][2] if x in rgb_mapping else None
    )
    temp_df["idx"] = temp_df["idx"].map(id_mapping)


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
    label_df = _combine_reports(
        per_section_df, atlas_labels, derive_fn=_derive_area_fractions
    )
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
    # Fill numeric columns with 0
    for col in [
        "pixel_count",
        "region_area",
        "left_hemi_pixel_count",
        "left_hemi_region_area",
        "right_hemi_pixel_count",
        "right_hemi_region_area",
    ]:
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


def _backfill_label_info(label_df, atlas_labels):
    """Fill missing name/colour columns from *atlas_labels* for zero-count regions."""
    if "name" not in atlas_labels.columns or "name" not in label_df.columns:
        return
    idx_map = atlas_labels.set_index("idx")
    for col in ["name", "r", "g", "b"]:
        if col in atlas_labels.columns:
            label_df[col] = label_df[col].fillna(label_df["idx"].map(idx_map[col]))


def _prepare_combined_df(per_section_df, available_group_cols):
    """Concat per-section DataFrames and coerce non-group columns to numeric."""
    non_empty = [df for df in per_section_df if not df.empty]
    combined = pd.concat(non_empty) if non_empty else pd.DataFrame()
    for col in combined.columns:
        if col not in available_group_cols:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def _combine_reports(per_section_df, atlas_labels, *, derive_fn):
    """Combine per-section DataFrames into a whole-series report.

    This is the single implementation shared by both the segmentation
    (area-fraction) and intensity (mean-intensity) pipelines.

    Args:
        per_section_df: List of per-section DataFrames.
        atlas_labels: Atlas labels DataFrame.
        derive_fn: Callable ``(df) → None`` that adds derived columns
            (e.g. area fractions or mean intensities) **in-place**.

    Returns:
        Combined DataFrame reindexed to all atlas regions.
    """
    group_cols = ["idx", "name", "r", "g", "b"]
    available_group_cols = [c for c in group_cols if c in per_section_df[0].columns]
    combined = _prepare_combined_df(per_section_df, available_group_cols)

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    sum_cols = [c for c in numeric_cols if c not in set(available_group_cols)]
    sum_cols = [c for c in sum_cols if c not in _RATIO_COLS]

    label_df = combined.groupby(available_group_cols)[sum_cols].sum().reset_index()

    derive_fn(label_df)
    label_df.fillna(0, inplace=True)
    label_df = reindex_to_atlas(label_df, atlas_labels)
    _backfill_label_info(label_df, atlas_labels)
    label_df.fillna(0, inplace=True)
    return label_df


def _derive_area_fractions(df):
    """Callback: add area-fraction columns to *df* in-place.

    Uses direct division to match the historical ``_combine_slice_reports``
    behaviour: ``inf`` for *x / 0* and ``NaN`` for *0 / 0*.  The caller
    follows up with ``fillna(0)`` which converts NaN → 0 and preserves
    inf (the expected output includes inf for regions with non-zero
    pixel counts but zero region area).
    """
    for num, den, res in AREA_FRACTION_PAIRS:
        if num in df.columns and den in df.columns:
            df[res] = df[num] / df[den]

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

from ...io.atlas_loader import resolve_atlas_labels
from .region_counting import pixel_count_per_region
from ..utils import (
    AREA_FRACTION_PAIRS,
    apply_area_fractions,
    apply_mean_intensities,
    reindex_to_atlas,
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
    rgb_series = temp_df["idx"].map(lambda x: rgb_mapping.get(x, (None, None, None)))
    rgb_df = pd.DataFrame(
        rgb_series.tolist(),
        index=temp_df.index,
        columns=["r", "g", "b"],
    )
    temp_df[["r", "g", "b"]] = rgb_df
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
    points_labels,
    centroids_labels,
    points_undamaged,
    centroids_undamaged,
    points_hemi,
    centroids_hemi,
    region_areas,
    atlas_labels,
    apply_damage_mask,
):
    """Aggregate per-pixel and per-centroid counts into a summary table.

    Args:
        points_labels: 1-D array of region IDs for points.
        centroids_labels: 1-D array of region IDs for centroids.
        points_undamaged: 1-D undamaged mask for points.
        centroids_undamaged: 1-D undamaged mask for centroids.
        points_hemi: 1-D hemisphere labels for points.
        centroids_hemi: 1-D hemisphere labels for centroids.
        region_areas: Combined region-area DataFrame (summed across sections).
        atlas_labels: Atlas labels DataFrame.
        apply_damage_mask: Whether damage mask was applied.

    Returns:
        label_df — whole-series DataFrame.
    """
    count_df = pixel_count_per_region(
        points_labels,
        centroids_labels,
        points_undamaged,
        centroids_undamaged,
        points_hemi,
        centroids_hemi,
        atlas_labels,
        apply_damage_mask,
    )
    label_df = _merge_dataframes(count_df, region_areas, atlas_labels)
    if not apply_damage_mask:
        cols = [c for c in label_df.columns if "damage" not in c]
        label_df = label_df[cols]
    return label_df


def _merge_dataframes(current_df, ra, atlas_labels):
    """Merge count DataFrame with region areas and atlas labels."""
    if ra is None or ra.empty:
        # No region areas — just merge counts with atlas labels.
        result = atlas_labels.merge(current_df, on="idx", how="left")
        result.fillna(0, inplace=True)
        apply_area_fractions(result)
        return result

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

    # Fill all numeric NaN values with 0 (counts, areas, damage, hemisphere cols).
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)

    apply_area_fractions(result)
    return result


# ── Intensity quantification ────────────────────────────────────────────


def quantify_intensity(region_intensities, atlas_labels):
    """Aggregate per-region intensity into a summary table.

    Args:
        region_intensities: Combined intensity DataFrame (already summed
            across sections by the batch processor).
        atlas_labels: Atlas labels DataFrame.

    Returns:
        label_df — whole-series DataFrame.
    """
    if region_intensities is None or region_intensities.empty:
        return pd.DataFrame()

    label_df = region_intensities.copy()
    apply_mean_intensities(label_df)
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
        label_df — whole-series DataFrame.
    """
    atlas_labels = resolve_atlas_labels(atlas_labels)

    if result.region_intensities is not None:
        return quantify_intensity(result.region_intensities, atlas_labels)

    return quantify_labeled_points(
        result.points.labels,
        result.objects.labels if result.objects is not None else np.array([], dtype=np.int64),
        result.points.undamaged_mask,
        result.objects.undamaged_mask if result.objects is not None else np.array([], dtype=bool),
        result.points.hemi_labels,
        result.objects.hemi_labels if result.objects is not None else np.array([], dtype=np.int64),
        result.region_areas,
        atlas_labels,
        apply_damage_mask,
    )

"""Per-region pixel and object counting.

Public API
----------
- :func:`pixel_count_per_region` — tally per-pixel/centroid counts by region.
"""

import numpy as np
import pandas as pd


def _empty_count_columns(with_hemisphere=False, with_damage=False):
    """Return the list of count-column names for an empty result DataFrame."""
    cols = ["idx", "name", "r", "g", "b", "pixel_count", "object_count"]
    if with_damage:
        cols += [
            "undamaged_object_count",
            "damaged_object_count",
            "undamaged_pixel_counts",
            "damaged_pixel_counts",
        ]
    if with_hemisphere:
        cols += [
            "left_hemi_pixel_count",
            "right_hemi_pixel_count",
            "left_hemi_object_count",
            "right_hemi_object_count",
        ]
    if with_damage and with_hemisphere:
        cols += [
            "left_hemi_undamaged_pixel_counts",
            "left_hemi_damaged_pixel_counts",
            "right_hemi_undamaged_pixel_counts",
            "right_hemi_damaged_pixel_counts",
            "left_hemi_undamaged_object_count",
            "left_hemi_damaged_object_count",
            "right_hemi_undamaged_object_count",
            "right_hemi_damaged_object_count",
        ]
    return cols


def _counts_for(
    mask: np.ndarray | None, arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (label_ids, counts) from *arr*, optionally filtered by boolean *mask*."""
    _empty = np.array([], dtype=np.int64)
    if arr.size == 0:
        return _empty, _empty

    data = np.asarray(arr if mask is None else arr[mask])
    if data.size == 0:
        return _empty, _empty

    if data.dtype != np.int64:
        data = data.astype(np.int64, copy=False)

    # Filter negative / background ids.
    data = data[data >= 0]
    if data.size == 0:
        return _empty, _empty

    max_id = int(data.max())

    # Fast path when label ids are reasonably bounded.
    if max_id <= 2_000_000:
        bc = np.bincount(data, minlength=max_id + 1)
        labels = np.nonzero(bc)[0].astype(np.int64, copy=False)
        return labels, bc[labels].astype(np.int64, copy=False)

    labels, counts = np.unique(data, return_counts=True)
    return labels.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def _lookup_counts(
    idx: np.ndarray, labels: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Map *idx* values to corresponding *counts* via sorted *labels*."""
    if idx.size == 0 or labels.size == 0:
        return np.zeros(idx.shape, dtype=np.int64)
    pos = np.searchsorted(labels, idx)
    out = np.zeros(idx.shape, dtype=np.int64)
    valid = (pos >= 0) & (pos < labels.size)
    if np.any(valid):
        pos_v = pos[valid]
        found = labels[pos_v] == idx[valid]
        if np.any(found):
            out_idx = np.flatnonzero(valid)[found]
            out[out_idx] = counts[pos_v[found]]
    return out


def _build_count_mask(hemi_arr, undamaged_arr, hemi_val, dmg_val):
    """Build a boolean mask combining hemisphere and damage filters."""
    mask = None
    if hemi_val is not None:
        mask = hemi_arr == hemi_val
    if dmg_val is not None:
        dmg_mask = undamaged_arr if dmg_val else ~undamaged_arr
        mask = dmg_mask if mask is None else (mask & dmg_mask)
    return mask


def _derive_count_aggregates(base, with_hemi, with_damage):
    """Derive aggregate columns (totals) from leaf-level count columns.

    When hemisphere data is present, grand totals are taken from the
    unfiltered ``_total_*`` columns (which include hemi=0 points) rather
    than summing left + right, so that points outside the hemisphere mask
    are not silently dropped.
    """
    if with_hemi and with_damage:
        for entity in ("pixel_count", "object_count"):
            base[f"left_hemi_{entity}"] = (
                base[f"left_hemi_undamaged_{entity}"]
                + base[f"left_hemi_damaged_{entity}"]
            )
            base[f"right_hemi_{entity}"] = (
                base[f"right_hemi_undamaged_{entity}"]
                + base[f"right_hemi_damaged_{entity}"]
            )
            base[f"undamaged_{entity}"] = base[f"_total_undamaged_{entity}"]
            base[f"damaged_{entity}"] = base[f"_total_damaged_{entity}"]
            base[entity] = base[f"undamaged_{entity}"] + base[f"damaged_{entity}"]
        base.drop(
            columns=[c for c in base.columns if c.startswith("_total_")],
            inplace=True,
        )
    elif with_hemi:
        for entity in ("pixel_count", "object_count"):
            base[entity] = base[f"_total_{entity}"]
        base.drop(
            columns=[c for c in base.columns if c.startswith("_total_")],
            inplace=True,
        )
    elif with_damage:
        for entity in ("pixel_count", "object_count"):
            base[entity] = base[f"undamaged_{entity}"] + base[f"damaged_{entity}"]
    # else: leaves are already named 'pixel_count' / 'object_count'

    # Legacy naming: damage-related pixel_count columns get trailing 's'
    renames = {
        c: c + "s"
        for c in base.columns
        if c.endswith("_pixel_count") and ("damaged" in c or "undamaged" in c)
    }
    if renames:
        base.rename(columns=renames, inplace=True)


def pixel_count_per_region(
    per_point_labels,
    per_centroid_labels,
    current_points_undamaged,
    current_centroids_undamaged,
    current_points_hemi,
    current_centroids_hemi,
    df_label_colours,
):
    """
    Tally object counts by region, tracking damage when a damage mask is present.

    Args:
        per_point_labels (ndarray): 1-D region labels for points.
        per_centroid_labels (ndarray): 1-D region labels for centroids.
        current_points_undamaged (ndarray or None): Undamaged-state flags for points,
            or None when no damage mask was applied.
        current_centroids_undamaged (ndarray or None): Undamaged-state flags for
            centroids, or None when no damage mask was applied.
        current_points_hemi (ndarray): Hemisphere tags for points.
        current_centroids_hemi (ndarray): Hemisphere tags for centroids.
        df_label_colours (DataFrame): Region label colors.

    Returns:
        DataFrame: Summed counts per region.
    """
    # Damage tracking: enabled when an undamaged mask is present.
    with_damage = current_points_undamaged is not None
    # If hemisphere labels are present, they are integers (1/2). If absent, they are None.
    with_hemi = None not in current_points_hemi

    # ── Build leaf count specs ───────────────────────────────────────
    hemi_iter = [(1, "left_hemi_"), (2, "right_hemi_")] if with_hemi else [(None, "")]
    dmg_iter = (
        [(True, "undamaged_"), (False, "damaged_")] if with_damage else [(None, "")]
    )

    computed = {}  # col_name -> (idx_array, count_array)
    all_indices = []

    for hemi_val, hemi_pfx in hemi_iter:
        for dmg_val, dmg_pfx in dmg_iter:
            p_mask = _build_count_mask(
                current_points_hemi, current_points_undamaged, hemi_val, dmg_val
            )
            c_mask = _build_count_mask(
                current_centroids_hemi, current_centroids_undamaged, hemi_val, dmg_val
            )

            px_col = f"{hemi_pfx}{dmg_pfx}pixel_count"
            obj_col = f"{hemi_pfx}{dmg_pfx}object_count"

            p_idx, p_cnt = _counts_for(p_mask, per_point_labels)
            c_idx, c_cnt = _counts_for(c_mask, per_centroid_labels)

            computed[px_col] = (p_idx, p_cnt)
            computed[obj_col] = (c_idx, c_cnt)
            all_indices.extend([p_idx, c_idx])

    # Compute unfiltered totals (no hemisphere filter) so that points with
    # hemi=0 (outside the hemisphere mask) are included in the grand totals.
    if with_hemi:
        for dmg_val, dmg_pfx in dmg_iter:
            p_mask = _build_count_mask(
                current_points_hemi, current_points_undamaged, None, dmg_val
            )
            c_mask = _build_count_mask(
                current_centroids_hemi, current_centroids_undamaged, None, dmg_val
            )
            p_idx, p_cnt = _counts_for(p_mask, per_point_labels)
            c_idx, c_cnt = _counts_for(c_mask, per_centroid_labels)
            computed[f"_total_{dmg_pfx}pixel_count"] = (p_idx, p_cnt)
            computed[f"_total_{dmg_pfx}object_count"] = (c_idx, c_cnt)
            all_indices.extend([p_idx, c_idx])

    # ── Build sparse DataFrame ────────────────────────────────────────
    all_idx = (
        np.unique(np.concatenate(all_indices))
        if all_indices
        else np.array([], dtype=np.int64)
    )
    if all_idx.size == 0:
        return pd.DataFrame(
            columns=_empty_count_columns(with_hemisphere=with_hemi, with_damage=with_damage)
        )

    base = df_label_colours[df_label_colours["idx"].isin(all_idx)].copy()

    # Preserve cells on atlas background (label 0) as "out_of_atlas".
    if 0 in all_idx and (base.empty or 0 not in base["idx"].values):
        oot_row = pd.DataFrame(
            {"idx": [0], "name": ["out_of_atlas"], "r": [0], "g": [0], "b": [0]}
        )
        base = pd.concat([base, oot_row], ignore_index=True)

    idx = base["idx"].to_numpy().astype(np.int64, copy=False)

    for col, (c_idx, c_cnt) in computed.items():
        base[col] = _lookup_counts(idx, c_idx, c_cnt)

    # ── Derive aggregate columns from leaves ──────────────────────────
    _derive_count_aggregates(base, with_hemi, with_damage)

    return base

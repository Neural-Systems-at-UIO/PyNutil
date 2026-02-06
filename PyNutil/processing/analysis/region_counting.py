"""Per-region pixel and object counting.

Public API
----------
- :func:`create_base_counts_dict` — skeleton dictionary for count accumulation.
- :func:`pixel_count_per_region` — tally per-pixel/centroid counts by region.
"""

import numpy as np
import pandas as pd


def create_base_counts_dict(with_hemisphere=False, with_damage=False):
    """
    Creates and returns a base dictionary structure for tracking counts.

    Args:
        with_hemisphere (bool): If True, include hemisphere fields.
        with_damage (bool): If True, include damage fields.

    Returns:
        dict: Structure containing count lists for pixels/objects.
    """
    counts = {
        "idx": [],
        "name": [],
        "r": [],
        "g": [],
        "b": [],
        "pixel_count": [],
        "object_count": [],
    }
    if with_damage:
        damage_fields = {
            "undamaged_object_count": [],
            "damaged_object_count": [],
            "undamaged_pixel_count": [],
            "damaged_pixel_counts": [],
        }
        counts.update(damage_fields)
    if with_hemisphere:
        hemisphere_fields = {
            "left_hemi_pixel_count": [],
            "right_hemi_pixel_count": [],
            "left_hemi_object_count": [],
            "right_hemi_object_count": [],
        }
        counts.update(hemisphere_fields)
    if with_damage and with_hemisphere:
        damage_hemisphere_fields = {
            "left_hemi_undamaged_pixel_count": [],
            "left_hemi_damaged_pixel_count": [],
            "right_hemi_undamaged_pixel_count": [],
            "right_hemi_damaged_pixel_count": [],
            "left_hemi_undamaged_object_count": [],
            "left_hemi_damaged_object_count": [],
            "right_hemi_undamaged_object_count": [],
            "right_hemi_damaged_object_count": [],
        }
        counts.update(damage_hemisphere_fields)
    return counts


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
    """Derive aggregate columns (totals) from leaf-level count columns."""
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
            base[f"undamaged_{entity}"] = (
                base[f"left_hemi_undamaged_{entity}"]
                + base[f"right_hemi_undamaged_{entity}"]
            )
            base[f"damaged_{entity}"] = (
                base[f"left_hemi_damaged_{entity}"]
                + base[f"right_hemi_damaged_{entity}"]
            )
            base[entity] = base[f"undamaged_{entity}"] + base[f"damaged_{entity}"]
    elif with_hemi:
        for entity in ("pixel_count", "object_count"):
            base[entity] = base[f"left_hemi_{entity}"] + base[f"right_hemi_{entity}"]
    elif with_damage:
        for entity in ("pixel_count", "object_count"):
            base[entity] = base[f"undamaged_{entity}"] + base[f"damaged_{entity}"]
    # else: leaves are already named 'pixel_count' / 'object_count'

    # Legacy naming: "damaged_pixel_counts" (trailing 's')
    if "damaged_pixel_count" in base.columns:
        base.rename(
            columns={"damaged_pixel_count": "damaged_pixel_counts"}, inplace=True
        )


def pixel_count_per_region(
    labels_dict_points,
    labeled_dict_centroids,
    current_points_undamaged,
    current_centroids_undamaged,
    current_points_hemi,
    current_centroids_hemi,
    df_label_colours,
    with_damage=False,
):
    """
    Tally object counts by region, optionally tracking damage and hemispheres.

    Args:
        labels_dict_points (dict): Maps points to region labels.
        labeled_dict_centroids (dict): Maps centroids to region labels.
        current_points_undamaged (ndarray): Undamaged-state flags for points.
        current_centroids_undamaged (ndarray): Undamaged-state flags for centroids.
        current_points_hemi (ndarray): Hemisphere tags for points.
        current_centroids_hemi (ndarray): Hemisphere tags for centroids.
        df_label_colours (DataFrame): Region label colors.
        with_damage (bool, optional): Track damage counts if True.

    Returns:
        DataFrame: Summed counts per region.
    """
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

            p_idx, p_cnt = _counts_for(p_mask, labels_dict_points)
            c_idx, c_cnt = _counts_for(c_mask, labeled_dict_centroids)

            computed[px_col] = (p_idx, p_cnt)
            computed[obj_col] = (c_idx, c_cnt)
            all_indices.extend([p_idx, c_idx])

    # ── Build sparse DataFrame ────────────────────────────────────────
    all_idx = (
        np.unique(np.concatenate(all_indices))
        if all_indices
        else np.array([], dtype=np.int64)
    )
    if all_idx.size == 0:
        return pd.DataFrame(
            columns=list(
                create_base_counts_dict(
                    with_hemisphere=with_hemi, with_damage=with_damage
                ).keys()
            )
        )

    base = df_label_colours[df_label_colours["idx"].isin(all_idx)].copy()
    idx = base["idx"].to_numpy().astype(np.int64, copy=False)

    for col, (c_idx, c_cnt) in computed.items():
        base[col] = _lookup_counts(idx, c_idx, c_cnt)

    # ── Derive aggregate columns from leaves ──────────────────────────
    _derive_count_aggregates(base, with_hemi, with_damage)

    return base

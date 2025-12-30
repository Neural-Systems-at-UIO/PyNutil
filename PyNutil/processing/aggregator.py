from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def build_region_intensity_dataframe(
    *,
    atlas_map: np.ndarray,
    intensity_resized: np.ndarray,
    atlas_labels: pd.DataFrame,
    region_areas: pd.DataFrame,
    hemi_mask: Optional[np.ndarray] = None,
    damage_mask_resized: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Aggregate intensity per atlas region.

    Mirrors the logic previously in `segmentation_to_atlas_space_intensity`.
    """

    flat_labels = atlas_map.ravel()
    flat_intensity = intensity_resized.ravel()

    unique_labels, inverse_indices = np.unique(flat_labels, return_inverse=True)
    sums = np.bincount(inverse_indices, weights=flat_intensity)
    counts = np.bincount(inverse_indices)

    df = pd.DataFrame({"idx": unique_labels, "sum_intensity": sums, "pixel_count": counts})

    if hemi_mask is not None:
        for hemi_id, hemi_name in [(1, "left_hemi"), (2, "right_hemi")]:
            mask = hemi_mask == hemi_id
            if damage_mask_resized is not None:
                mask &= damage_mask_resized

            h_labels = atlas_map[mask]
            h_intensity = intensity_resized[mask]

            if h_labels.size > 0:
                h_unique, h_inverse = np.unique(h_labels, return_inverse=True)
                h_sums = np.bincount(h_inverse, weights=h_intensity)
                h_counts = np.bincount(h_inverse)

                h_df = pd.DataFrame(
                    {
                        "idx": h_unique,
                        f"{hemi_name}_sum_intensity": h_sums,
                        f"{hemi_name}_pixel_count": h_counts,
                    }
                )
                df = df.merge(h_df, on="idx", how="left").fillna(0)
            else:
                df[f"{hemi_name}_sum_intensity"] = 0.0
                df[f"{hemi_name}_pixel_count"] = 0

    df = df.merge(region_areas, on="idx", how="left")
    df = df.merge(atlas_labels[["idx", "name", "r", "g", "b"]], on="idx", how="left")
    return df

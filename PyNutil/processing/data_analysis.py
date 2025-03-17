import pandas as pd
from .counting_and_load import pixel_count_per_region
import numpy as np


def map_to_custom_regions(custom_regions_dict, points_labels):
    custom_points_labels = np.zeros_like(points_labels)
    for i in np.unique(points_labels):
        new_id = np.where([i in r for r in custom_regions_dict["subregion_ids"]])[0]
        if len(new_id) > 1:
            raise ValueError(f"error, region id {i} is in more than one custom region")
        if len(new_id) == 0:
            continue
        new_id = new_id[0]
        new_id = custom_regions_dict["custom_ids"][new_id]
        custom_points_labels[points_labels == i] = int(new_id)
    return custom_points_labels


def apply_custom_regions(df, custom_regions_dict):
    # Create mappings
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

    # Update the original df with new columns
    df["custom_region_name"] = df["idx"].map(name_mapping).fillna("")
    temp_df = df.copy()

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

    # Group and aggregate
    grouped_df = (
        temp_df[temp_df["custom_region_name"] != ""]
        .groupby("custom_region_name", dropna=True)
        .agg(
            {
                "pixel_count": "sum",
                "undamaged_pixel_count": "sum",
                "damaged_pixel_counts": "sum",
                "region_area": "sum",
                "undamaged_region_area": "sum",
                "damaged_region_area": "sum",
                "object_count": "sum",
                "undamaged_object_count": "sum",
                "damaged_object_count": "sum",
                "r": "first",
                "g": "first",
                "b": "first",
            }
        )
        .reset_index()
    )

    grouped_df = grouped_df.rename(columns={"custom_region_name": "name"})

    grouped_df["area_fraction"] = grouped_df["pixel_count"] / grouped_df["region_area"]
    grouped_df["undamaged_area_fraction"] = (
        grouped_df["undamaged_pixel_count"] / grouped_df["undamaged_region_area"]
    )
    common_columns = [col for col in df.columns if col in grouped_df.columns]
    grouped_df = grouped_df.reindex(
        columns=common_columns
        + [col for col in grouped_df.columns if col not in common_columns]
    )
    return grouped_df, df


def quantify_labeled_points(
    points_len,
    centroids_len,
    region_areas_list,
    labeled_points,
    labeled_points_centroids,
    atlas_labels,
    per_point_undamaged,
    per_centroid_undamaged,
):
    """
    Quantifies labeled points and returns various DataFrames.

    Args:
        pixel_points (ndarray): Array of pixel points.
        centroids (ndarray): Array of centroids.
        points_len (list): List of lengths of points per section.
        centroids_len (list): List of lengths of centroids per section.
        region_areas_list (list): List of region areas per section.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        atlas_volume (ndarray): Volume with atlas labels.

    Returns:
        tuple: Labeled points, labeled centroids, label DataFrame, per section DataFrame.
    """
    # labeled_points_centroids = label_points(centroids, atlas_volume)
    # labeled_points = label_points(pixel_points, atlas_volume, scale_factor=1)

    per_section_df = _quantify_per_section(
        labeled_points,
        labeled_points_centroids,
        points_len,
        centroids_len,
        region_areas_list,
        atlas_labels,
        per_point_undamaged,
        per_centroid_undamaged,
    )
    label_df = _combine_slice_reports(per_section_df, atlas_labels)

    return label_df, per_section_df


def _quantify_per_section(
    labeled_points,
    labeled_points_centroids,
    points_len,
    centroids_len,
    region_areas_list,
    atlas_labels,
    per_point_undamaged,
    per_centroid_undamaged,
):
    """
    Quantifies labeled points per section.

    Args:
        labeled_points (ndarray): Array of labeled points.
        labeled_points_centroids (ndarray): Array of labeled centroids.
        points_len (list): List of lengths of points per section.
        centroids_len (list): List of lengths of centroids per section.
        region_areas_list (list): List of region areas per section.
        atlas_labels (DataFrame): DataFrame with atlas labels.

    Returns:
        list: List of DataFrames for each section.
    """
    prev_pl = 0
    prev_cl = 0
    per_section_df = []

    for pl, cl, ra in zip(points_len, centroids_len, region_areas_list):
        current_centroids = labeled_points_centroids[prev_cl : prev_cl + cl]
        current_points = labeled_points[prev_pl : prev_pl + pl]
        current_points_undamaged = per_point_undamaged[prev_pl : prev_pl + pl]
        current_centroids_undamaged = per_centroid_undamaged[prev_cl : prev_cl + cl]
        current_df = pixel_count_per_region(
            current_points,
            current_centroids,
            current_points_undamaged,
            current_centroids_undamaged,
            atlas_labels,
        )
        current_df_new = _merge_dataframes(current_df, ra, atlas_labels)
        per_section_df.append(current_df_new)
        prev_pl += pl
        prev_cl += cl

    return per_section_df


def _merge_dataframes(current_df, ra, atlas_labels):
    """
    Merges current DataFrame with region areas and atlas labels.

    Args:
        current_df (DataFrame): Current DataFrame.
        ra (DataFrame): DataFrame with region areas.
        atlas_labels (DataFrame): DataFrame with atlas labels.

    Returns:
        DataFrame: Merged DataFrame.
    """
    all_region_df = atlas_labels.merge(ra, on="idx", how="left")
    current_df_new = all_region_df.merge(
        current_df, on="idx", how="left", suffixes=(None, "_y")
    ).drop(columns=["name_y", "r_y", "g_y", "b_y"])
    current_df_new["area_fraction"] = (
        current_df_new["pixel_count"] / current_df_new["region_area"]
    )
    current_df_new.fillna(0, inplace=True)
    return current_df_new


def _combine_slice_reports(per_section_df, atlas_labels):
    """
    Combines slice reports into a single DataFrame.

    Args:
        per_section_df (list): List of DataFrames for each section.
        atlas_labels (DataFrame): DataFrame with atlas labels.

    Returns:
        DataFrame: Combined DataFrame.
    """
    label_df = (
        pd.concat(per_section_df)
        .groupby(["idx", "name", "r", "g", "b"])
        .sum()
        .reset_index()
        .drop(columns=["area_fraction"])
    )
    label_df["area_fraction"] = label_df["pixel_count"] / label_df["region_area"]
    label_df["undamaged_area_fraction"] = (
        label_df["undamaged_pixel_count"] / label_df["undamaged_region_area"]
    )
    label_df.fillna(0, inplace=True)

    label_df = label_df.set_index("idx")
    label_df = label_df.reindex(index=atlas_labels["idx"])
    label_df = label_df.reset_index()
    return label_df

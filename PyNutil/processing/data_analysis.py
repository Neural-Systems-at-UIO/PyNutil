import pandas as pd
from .counting_and_load import pixel_count_per_region
import numpy as np


def map_to_custom_regions(custom_regions_dict, points_labels):
    """
    Reassigns atlas-region labels into user-defined custom regions.

    Args:
        atlas_labeled_points (DataFrame): DataFrame of points, each with atlas labels.
        custom_region_map (dict): Mapping of atlas region IDs to custom IDs.

    Returns:
        DataFrame: Points with updated region assignments.
    """
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
    """
    Applies a custom region definition to the image's region labels.

    Args:
        image (ndarray): The image array whose regions are being remapped.
        custom_region_file (str): File path or identifier for custom region definitions.

    Returns:
        ndarray: The image with modified region labels.
    """
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

    # Define all possible columns to aggregate
    possible_columns = [
        "pixel_count",
        "undamaged_pixel_count",
        "damaged_pixel_counts",
        "region_area",
        "undamaged_region_area",
        "damaged_region_area",
        "object_count",
        "undamaged_object_count",
        "damaged_object_count",
        "left_hemi_pixel_count",
        "left_hemi_undamaged_pixel_count",
        "left_hemi_damaged_pixel_count",
        "left_hemi_region_area",
        "left_hemi_undamaged_region_area",
        "left_hemi_damaged_region_area",
        "left_hemi_object_count",
        "left_hemi_undamaged_object_count",
        "left_hemi_damaged_object_count",
        "right_hemi_pixel_count",
        "right_hemi_undamaged_pixel_count",
        "right_hemi_damaged_pixel_count",
        "right_hemi_region_area",
        "right_hemi_undamaged_region_area",
        "right_hemi_damaged_region_area",
        "right_hemi_object_count",
        "right_hemi_undamaged_object_count",
        "right_hemi_damaged_object_count",
    ]

    # Only include columns that actually exist in the DataFrame
    agg_dict = {col: "sum" for col in possible_columns if col in temp_df.columns}
    # Add the color columns
    agg_dict.update({"r": "first", "g": "first", "b": "first"})

    # Group and aggregate only existing columns
    grouped_df = (
        temp_df[temp_df["custom_region_name"] != ""]
        .groupby("custom_region_name", dropna=True)
        .agg(agg_dict)
        .reset_index()
    )

    grouped_df = grouped_df.rename(columns={"custom_region_name": "name"})

    # Calculate area fractions only if required columns exist
    if "pixel_count" in grouped_df and "region_area" in grouped_df:
        grouped_df["area_fraction"] = (
            grouped_df["pixel_count"] / grouped_df["region_area"]
        )

    if "undamaged_pixel_count" in grouped_df and "undamaged_region_area" in grouped_df:
        grouped_df["undamaged_area_fraction"] = (
            grouped_df["undamaged_pixel_count"] / grouped_df["undamaged_region_area"]
        )

    if "left_hemi_pixel_count" in grouped_df and "left_hemi_region_area" in grouped_df:
        grouped_df["left_hemi_area_fraction"] = (
            grouped_df["left_hemi_pixel_count"] / grouped_df["left_hemi_region_area"]
        )

    if (
        "right_hemi_pixel_count" in grouped_df
        and "right_hemi_region_area" in grouped_df
    ):
        grouped_df["right_hemi_area_fraction"] = (
            grouped_df["right_hemi_pixel_count"] / grouped_df["right_hemi_region_area"]
        )

    if (
        "left_hemi_undamaged_pixel_count" in grouped_df
        and "left_hemi_undamaged_region_area" in grouped_df
    ):
        grouped_df["left_hemi_undamaged_area_fraction"] = (
            grouped_df["left_hemi_undamaged_pixel_count"]
            / grouped_df["left_hemi_undamaged_region_area"]
        )

    if (
        "right_hemi_undamaged_pixel_count" in grouped_df
        and "right_hemi_undamaged_region_area" in grouped_df
    ):
        grouped_df["right_hemi_undamaged_area_fraction"] = (
            grouped_df["right_hemi_undamaged_pixel_count"]
            / grouped_df["right_hemi_undamaged_region_area"]
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
    points_hemi_labels,
    centroids_hemi_labels,
    per_point_undamaged,
    per_centroid_undamaged,
    apply_damage_mask,
):
    """
    Aggregates labeled points into a summary table.

    Args:
        points (ndarray): Array of point coordinates and labels.
        atlas_labels (DataFrame): DataFrame containing atlas region labels.

    Returns:
        DataFrame: Summarized point counts per region.
    """
    per_section_df = _quantify_per_section(
        labeled_points,
        labeled_points_centroids,
        points_len,
        centroids_len,
        region_areas_list,
        atlas_labels,
        per_point_undamaged,
        per_centroid_undamaged,
        points_hemi_labels,
        centroids_hemi_labels,
        apply_damage_mask,
    )
    label_df = _combine_slice_reports(per_section_df, atlas_labels)
    if not apply_damage_mask:
        cols = [i for i in label_df.columns if "damage" not in i]
        label_df = label_df[cols]
        per_section_df = [i[cols] for i in per_section_df]
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
    points_hemi_labels,
    centroids_hemi_labels,
    with_damage=False,
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
        current_points_hemi = points_hemi_labels[prev_pl : prev_pl + pl]
        current_centroids_hemi = centroids_hemi_labels[prev_cl : prev_cl + cl]
        current_df = pixel_count_per_region(
            current_points,
            current_centroids,
            current_points_undamaged,
            current_centroids_undamaged,
            current_points_hemi,
            current_centroids_hemi,
            atlas_labels,
            with_damage,
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
    cols_to_use = ra.columns.difference(atlas_labels.columns)
    all_region_df = atlas_labels.merge(ra[["idx", *cols_to_use]], on="idx", how="left")
    cols_to_use = current_df.columns.difference(all_region_df.columns)
    current_df_new = all_region_df.merge(
        current_df[["idx", *cols_to_use]], on="idx", how="left"
    )
    if (
        "pixel_count" in current_df_new.columns
        and "region_area" in current_df_new.columns
    ):
        current_df_new["area_fraction"] = (
            current_df_new["pixel_count"] / current_df_new["region_area"]
        )
    if (
        "left_hemi_pixel_count" in current_df_new.columns
        and "left_hemi_region_area" in current_df_new.columns
    ):
        current_df_new["left_hemi_area_fraction"] = (
            current_df_new["left_hemi_pixel_count"]
            / current_df_new["left_hemi_region_area"]
        )
    if (
        "right_hemi_pixel_count" in current_df_new.columns
        and "right_hemi_region_area" in current_df_new.columns
    ):
        current_df_new["right_hemi_area_fraction"] = (
            current_df_new["right_hemi_pixel_count"]
            / current_df_new["right_hemi_region_area"]
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
    if "left_hemi_pixel_count" in label_df:
        label_df["left_hemi_area_fraction"] = (
            label_df["left_hemi_pixel_count"] / label_df["left_hemi_region_area"]
        )
        label_df["right_hemi_area_fraction"] = (
            label_df["right_hemi_pixel_count"] / label_df["right_hemi_region_area"]
        )
    if "undamaged_region_area" in label_df:
        label_df["undamaged_area_fraction"] = (
            label_df["undamaged_pixel_count"] / label_df["undamaged_region_area"]
        )
    if ("left_hemi_pixel_count" in label_df) and ("undamaged_region_area" in label_df):
        label_df["left_hemi_undamaged_area_fraction"] = (
            label_df["left_hemi_undamaged_pixel_count"]
            / label_df["left_hemi_undamaged_region_area"]
        )
        label_df["right_hemi_undamaged_area_fraction"] = (
            label_df["right_hemi_undamaged_pixel_count"]
            / label_df["right_hemi_undamaged_region_area"]
        )
    label_df.fillna(0, inplace=True)
    label_df = label_df.set_index("idx")
    label_df = label_df.reindex(index=atlas_labels["idx"])
    label_df = label_df.reset_index()
    return label_df

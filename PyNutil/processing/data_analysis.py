import pandas as pd
from .counting_and_load import pixel_count_per_region, label_points


def quantify_labeled_points(
    pixel_points,
    centroids,
    points_len,
    centroids_len,
    region_areas_list,
    atlas_labels,
    atlas_volume,
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
    labeled_points_centroids = label_points(centroids, atlas_volume)
    labeled_points = label_points(pixel_points, atlas_volume, scale_factor=1)

    per_section_df = _quantify_per_section(
        labeled_points,
        labeled_points_centroids,
        points_len,
        centroids_len,
        region_areas_list,
        atlas_labels,
    )
    label_df = _combine_slice_reports(per_section_df, atlas_labels)

    return labeled_points, labeled_points_centroids, label_df, per_section_df


def _quantify_per_section(
    labeled_points,
    labeled_points_centroids,
    points_len,
    centroids_len,
    region_areas_list,
    atlas_labels,
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
        current_df = pixel_count_per_region(
            current_points, current_centroids, atlas_labels
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
    label_df.fillna(0, inplace=True)

    label_df = label_df.set_index("idx")
    label_df = label_df.reindex(index=atlas_labels["idx"])
    label_df = label_df.reset_index()
    return label_df

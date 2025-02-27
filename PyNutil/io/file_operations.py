import os
from .read_and_write import read_atlas_volume, write_points_to_meshview


def save_analysis_output(
    pixel_points,
    centroids,
    label_df,
    per_section_df,
    labeled_points,
    labeled_points_centroids,
    points_len,
    centroids_len,
    segmentation_filenames,
    atlas_labels,
    output_folder,
):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/whole_series_report", exist_ok=True)
    os.makedirs(f"{output_folder}/per_section_meshview", exist_ok=True)
    os.makedirs(f"{output_folder}/per_section_reports", exist_ok=True)
    os.makedirs(f"{output_folder}/whole_series_meshview", exist_ok=True)

    if label_df is not None:
        label_df.to_csv(
            f"{output_folder}/whole_series_report/counts.csv",
            sep=";",
            na_rep="",
            index=False,
        )
    else:
        print("No quantification found, so only coordinates will be saved.")
        print(
            "If you want to save the quantification, please run quantify_coordinates."
        )

    _save_per_section_reports(
        per_section_df,
        segmentation_filenames,
        points_len,
        centroids_len,
        pixel_points,
        centroids,
        labeled_points,
        labeled_points_centroids,
        atlas_labels,
        output_folder,
    )
    _save_whole_series_meshview(
        pixel_points,
        labeled_points,
        centroids,
        labeled_points_centroids,
        atlas_labels,
        output_folder,
    )


def _save_per_section_reports(
    per_section_df,
    segmentation_filenames,
    points_len,
    centroids_len,
    pixel_points,
    centroids,
    labeled_points,
    labeled_points_centroids,
    atlas_labels,
    output_folder,
):
    prev_pl = 0
    prev_cl = 0

    for pl, cl, fn, df in zip(
        points_len,
        centroids_len,
        segmentation_filenames,
        per_section_df,
    ):
        split_fn = fn.split(os.sep)[-1].split(".")[0]
        df.to_csv(
            f"{output_folder}/per_section_reports/{split_fn}.csv",
            sep=";",
            na_rep="",
            index=False,
        )
        _save_per_section_meshview(
            split_fn,
            pl,
            cl,
            prev_pl,
            prev_cl,
            pixel_points,
            centroids,
            labeled_points,
            labeled_points_centroids,
            atlas_labels,
            output_folder,
        )
        prev_cl += cl
        prev_pl += pl


def _save_per_section_meshview(
    split_fn,
    pl,
    cl,
    prev_pl,
    prev_cl,
    pixel_points,
    centroids,
    labeled_points,
    labeled_points_centroids,
    atlas_labels,
    output_folder,
):
    write_points_to_meshview(
        pixel_points[prev_pl : pl + prev_pl],
        labeled_points[prev_pl : pl + prev_pl],
        f"{output_folder}/per_section_meshview/{split_fn}_pixels.json",
        atlas_labels,
    )
    write_points_to_meshview(
        centroids[prev_cl : cl + prev_cl],
        labeled_points_centroids[prev_cl : cl + prev_cl],
        f"{output_folder}/per_section_meshview/{split_fn}_centroids.json",
        atlas_labels,
    )


def _save_whole_series_meshview(
    pixel_points,
    labeled_points,
    centroids,
    labeled_points_centroids,
    atlas_labels,
    output_folder,
):
    write_points_to_meshview(
        pixel_points,
        labeled_points,
        f"{output_folder}/whole_series_meshview/pixels_meshview.json",
        atlas_labels,
    )
    write_points_to_meshview(
        centroids,
        labeled_points_centroids,
        f"{output_folder}/whole_series_meshview/objects_meshview.json",
        atlas_labels,
    )

import os
import json
from .read_and_write import write_hemi_points_to_meshview


def _ensure_analysis_output_dirs(output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for subdir in (
        "whole_series_report",
        "per_section_meshview",
        "per_section_reports",
        "whole_series_meshview",
    ):
        os.makedirs(f"{output_folder}/{subdir}", exist_ok=True)


def save_analysis_output(
    pixel_points,
    centroids,
    label_df,
    per_section_df,
    labeled_points,
    labeled_points_centroids,
    points_hemi_labels,
    centroids_hemi_labels,
    points_len,
    centroids_len,
    segmentation_filenames,
    atlas_labels,
    output_folder,
    segmentation_folder=None,
    image_folder=None,
    alignment_json=None,
    colour=None,
    intensity_channel=None,
    atlas_name=None,
    custom_region_path=None,
    atlas_path=None,
    label_path=None,
    settings_file=None,
    prepend=None,
    point_intensities=None,
    **kwargs,
):
    """
    Save the analysis output to the specified folder.

    Parameters
    ----------
    # ...existing code...
    output_folder : str
        The folder where the output will be saved.
    segmentation_folder : str, optional
        The folder containing the segmentation files (default is None).
    image_folder : str, optional
        The folder containing the original images (default is None).
    alignment_json : str, optional
        The path to the alignment JSON file (default is None).
    colour : list, optional
        The RGB colour of the object to be quantified in the segmentation (default is None).
    intensity_channel : str, optional
        The channel used for intensity quantification (default is None).
    atlas_name : str, optional
        The name of the atlas in the brainglobe api to be used for quantification (default is None).
    atlas_path : str, optional
        The path to the custom atlas volume file, only specific if you don't want to use brainglobe (default is None).
    label_path : str, optional
        The path to the custom atlas label file, only specific if you don't want to use brainglobe (default is None).
    settings_file : str, optional
        The path to the settings file that was used (default is None).
    """
    # Create the output folder if it doesn't exist
    _ensure_analysis_output_dirs(output_folder)
    # Filter out rows where 'region_area' is 0 in label_df
    # if label_df is not None and "region_area" in label_df.columns:
    #     label_df = label_df[label_df["region_area"] != 0]
    if label_df is not None:
        report_name = "intensity.csv" if image_folder else "counts.csv"
        label_df.to_csv(
            f"{output_folder}/whole_series_report/{prepend}{report_name}",
            sep=";",
            na_rep="",
            index=False,
        )
    elif not prepend:
        print("No quantification found, so only coordinates will be saved.")
        print(
            "If you want to save the quantification, please run quantify_coordinates."
        )

    if per_section_df is not None and segmentation_filenames is not None:
        _save_per_section_reports(
            per_section_df,
            segmentation_filenames,
            points_len,
            centroids_len,
            pixel_points,
            centroids,
            labeled_points,
            labeled_points_centroids,
            points_hemi_labels,
            centroids_hemi_labels,
            atlas_labels,
            output_folder,
            prepend,
            point_intensities,
            colormap=kwargs.get("colormap", "gray"),
        )
    if pixel_points is not None:
        _save_whole_series_meshview(
            pixel_points,
            labeled_points,
            centroids,
            labeled_points_centroids,
            points_hemi_labels,
            centroids_hemi_labels,
            atlas_labels,
            output_folder,
            prepend,
            point_intensities,
            colormap=kwargs.get("colormap", "gray"),
        )

    # Save settings to JSON file for reference
    settings_dict = {
        "segmentation_folder": segmentation_folder,
        "image_folder": image_folder,
        "alignment_json": alignment_json,
        "colour": colour,
        "intensity_channel": intensity_channel,
        "custom_region_path": custom_region_path,
    }

    # Add atlas information to settings
    if atlas_name:
        settings_dict["atlas_name"] = atlas_name
    if atlas_path:
        settings_dict["atlas_path"] = atlas_path
    if label_path:
        settings_dict["label_path"] = label_path
    if settings_file:
        settings_dict["settings_file"] = settings_file

    # Write settings to file
    settings_file_path = os.path.join(output_folder, "pynutil_settings.json")
    with open(settings_file_path, "w") as f:
        json.dump(settings_dict, f, indent=4)


def _save_per_section_reports(
    per_section_df,
    segmentation_filenames,
    points_len,
    centroids_len,
    pixel_points,
    centroids,
    labeled_points,
    labeled_points_centroids,
    points_hemi_labels,
    centroids_hemi_labels,
    atlas_labels,
    output_folder,
    prepend,
    point_intensities=None,
    colormap="gray",
):
    prev_pl = 0
    prev_cl = 0

    # Handle None for points_len and centroids_len (e.g. in intensity mode)
    if points_len is None:
        points_len = [0] * len(segmentation_filenames)
    if centroids_len is None:
        centroids_len = [0] * len(segmentation_filenames)

    for pl, cl, fn, df in zip(
        points_len,
        centroids_len,
        segmentation_filenames,
        per_section_df,
    ):
        split_fn = fn.split(os.sep)[-1].split(".")[0]
        df.to_csv(
            f"{output_folder}/per_section_reports/{prepend}{split_fn}.csv",
            sep=";",
            na_rep="",
            index=False,
        )
        if pixel_points is not None or centroids is not None:
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
                points_hemi_labels,
                centroids_hemi_labels,
                atlas_labels,
                output_folder,
                prepend,
                point_intensities[prev_pl : pl + prev_pl]
                if point_intensities is not None
                else None,
                colormap=colormap,
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
    points_hemi_labels,
    centroids_hemi_labels,
    atlas_labels,
    output_folder,
    prepend,
    point_intensities=None,
    colormap="gray",
):
    write_hemi_points_to_meshview(
        pixel_points[prev_pl : pl + prev_pl] if pixel_points is not None else None,
        labeled_points[prev_pl : pl + prev_pl] if labeled_points is not None else None,
        points_hemi_labels[prev_pl : pl + prev_pl]
        if points_hemi_labels is not None
        else None,
        f"{output_folder}/per_section_meshview/{prepend}{split_fn}_pixels.json",
        atlas_labels,
        point_intensities,
        colormap=colormap,
    )
    if centroids is not None:
        write_hemi_points_to_meshview(
            centroids[prev_cl : cl + prev_cl],
            labeled_points_centroids[prev_cl : cl + prev_cl],
            centroids_hemi_labels[prev_cl : cl + prev_cl],
            f"{output_folder}/per_section_meshview/{prepend}{split_fn}_centroids.json",
            atlas_labels,
            colormap=colormap,
        )


def _save_whole_series_meshview(
    pixel_points,
    labeled_points,
    centroids,
    labeled_points_centroids,
    points_hemi_labels,
    centroids_hemi_labels,
    atlas_labels,
    output_folder,
    prepend,
    point_intensities=None,
    colormap="gray",
):
    write_hemi_points_to_meshview(
        pixel_points,
        labeled_points,
        points_hemi_labels,
        f"{output_folder}/whole_series_meshview/{prepend}pixels_meshview.json",
        atlas_labels,
        point_intensities,
        colormap=colormap,
    )
    if centroids is not None:
        write_hemi_points_to_meshview(
            centroids,
            labeled_points_centroids,
            centroids_hemi_labels,
            f"{output_folder}/whole_series_meshview/{prepend}objects_meshview.json",
            atlas_labels,
            colormap=colormap,
        )

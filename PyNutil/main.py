import json
import os

from .io.atlas_loader import load_atlas_data, load_custom_atlas
from .data_analysis import quantify_labeled_points
from .io.file_operations import save_analysis_output
from .coordinate_extraction import folder_to_atlas_space
from .counting_and_load import label_points


class PyNutil:
    def __init__(
        self,
        segmentation_folder=None,
        alignment_json=None,
        colour=None,
        atlas_name=None,
        atlas_path=None,
        label_path=None,
        settings_file=None,
    ):
        if settings_file is not None:
            with open(settings_file, "r") as f:
                settings = json.load(f)
            try:
                segmentation_folder = settings["segmentation_folder"]
                alignment_json = settings["alignment_json"]
                colour = settings["colour"]
                if "atlas_path" in settings and "label_path" in settings:
                    atlas_path = settings["atlas_path"]
                    label_path = settings["label_path"]
                else:
                    atlas_name = settings["atlas_name"]
            except KeyError as exc:
                raise KeyError(
                    "Settings file must contain segmentation_folder, alignment_json, colour, and either atlas_path and label_path or atlas_name"
                ) from exc

        self.segmentation_folder = segmentation_folder
        self.alignment_json = alignment_json
        self.colour = colour
        self.atlas_name = atlas_name

        if (atlas_path or label_path) and atlas_name:
            raise ValueError(
                "Please specify either atlas_path and label_path or atlas_name. Atlas and label paths are only used for loading custom atlases."
            )

        if atlas_path and label_path:
            self.atlas_volume, self.atlas_labels = load_custom_atlas(atlas_path, label_path)
        else:
            self._check_atlas_name()
            self.atlas_volume, self.atlas_labels = load_atlas_data(atlas_name=atlas_name)

    def _check_atlas_name(self):
        if not self.atlas_name:
            raise ValueError("When atlas_path and label_path are not specified, atlas_name must be specified.")

    def get_coordinates(self, non_linear=True, object_cutoff=0, use_flat=False):
        """Extracts pixel coordinates from the segmentation data."""
        (
            self.pixel_points,
            self.centroids,
            self.region_areas_list,
            self.points_len,
            self.centroids_len,
            self.segmentation_filenames,
        ) = folder_to_atlas_space(
            self.segmentation_folder,
            self.alignment_json,
            self.atlas_labels,
            self.colour,
            non_linear,
            object_cutoff,
            self.atlas_volume,
            use_flat,
        )

    def quantify_coordinates(self):
        """Quantifies the pixel coordinates by region."""
        if not hasattr(self, "pixel_points") and not hasattr(self, "centroids"):
            raise ValueError("Please run get_coordinates before running quantify_coordinates.")

        self.labeled_points, self.labeled_points_centroids, self.label_df, self.per_section_df = quantify_labeled_points(
            self.pixel_points,
            self.centroids,
            self.points_len,
            self.centroids_len,
            self.region_areas_list,
            self.atlas_labels,
            self.atlas_volume
        )

    def save_analysis(self, output_folder):
        """Saves the pixel coordinates and pixel counts to different files in the specified output folder."""
        save_analysis_output(
            self.pixel_points,
            self.centroids,
            self.label_df,
            self.per_section_df,
            self.labeled_points,
            self.labeled_points_centroids,
            self.points_len,
            self.centroids_len,
            self.segmentation_filenames,
            self.atlas_labels,
            output_folder,
        )

import json
from .io.atlas_loader import load_atlas_data, load_custom_atlas
from .processing.data_analysis import quantify_labeled_points
from .io.file_operations import save_analysis_output
from .io.read_and_write import open_custom_region_file
from .processing.coordinate_extraction import folder_to_atlas_space
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
    temp_df["idx"] = temp_df["idx"].map(id_mapping)

    temp_df["r"] = temp_df["idx"].map(lambda x: rgb_mapping[x][0] if x in rgb_mapping else None)
    temp_df["g"] = temp_df["idx"].map(lambda x: rgb_mapping[x][1] if x in rgb_mapping else None)
    temp_df["b"] = temp_df["idx"].map(lambda x: rgb_mapping[x][2] if x in rgb_mapping else None)

    # Group and aggregate
    grouped_df = temp_df[temp_df["custom_region_name"] != ""].groupby("custom_region_name", dropna=True).agg({
        "pixel_count": "sum",
        "region_area": "sum",
        "object_count": "sum",
        "r": "first",
        "g": "first",
        "b": "first",
    }).reset_index()

    grouped_df = grouped_df.rename(columns={"custom_region_name": "name"})

    grouped_df["area_fraction"] = grouped_df["pixel_count"] / grouped_df["region_area"]
    common_columns = [col for col in df.columns if col in grouped_df.columns]
    grouped_df = grouped_df.reindex(columns=common_columns + [col for col in grouped_df.columns if col not in common_columns])
    return grouped_df, df

class PyNutil:
    """
    A class used to perform brain-wide quantification and spatial analysis of features in serial section images.

    Methods
    -------
    __init__(self, segmentation_folder=None, alignment_json=None, colour=None, atlas_name=None, atlas_path=None, label_path=None, settings_file=None)
        Initializes the PyNutil class with the given parameters.
    get_coordinates(self, non_linear=True, object_cutoff=0, use_flat=False)
        Extracts pixel coordinates from the segmentation data.
    quantify_coordinates(self)
        Quantifies the pixel coordinates by region.
    save_analysis(self, output_folder)
        Saves the pixel coordinates and pixel counts to different files in the specified output folder.
    """

    def __init__(
        self,
        segmentation_folder=None,
        alignment_json=None,
        colour=None,
        atlas_name=None,
        atlas_path=None,
        label_path=None,
        custom_region_path=None,
        settings_file=None,
    ):
        """
        Initializes the PyNutil class with the given parameters.

        Parameters
        ----------
        segmentation_folder : str, optional
            The folder containing the segmentation files (default is None).
        alignment_json : str, optional
            The path to the alignment JSON file (default is None).
        colour : list, optional
            The RGB colour of the object to be quantified in the segmentation (default is None).
        atlas_name : str, optional
            The name of the atlas in the brainglobe api to be used for quantification (default is None).
        atlas_path : str, optional
            The path to the custom atlas volume file, only specify if you don't want to use brainglobe (default is None).
        label_path : str, optional
            The path to the custom atlas label file, only specify if you don't want to use brainglobe (default is None).
        custom_region_path : str, optional
            The path to a custom region id file. This can be found
        settings_file : str, optional
            The path to the settings JSON file. This file contains the above parameters and is used for automation (default is None).

        Raises
        ------
        KeyError
            If the settings file does not contain the required keys.
        ValueError
            If both atlas_path and atlas_name are specified or if neither is specified.
        """
        try:
            if settings_file is not None:
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                try:
                    segmentation_folder = settings["segmentation_folder"]
                    alignment_json = settings["alignment_json"]
                    colour = settings["colour"]
                    if "custom_region_path" in settings:
                        custom_region_path = settings["custom_region_path"]
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
            self.custom_region_path = custom_region_path
            if custom_region_path:
                custom_regions_dict, custom_atlas_labels = open_custom_region_file(
                    custom_region_path
                )
            else:
                custom_regions_dict = None
                custom_atlas_labels = None
            self.custom_regions_dict = custom_regions_dict
            self.custom_atlas_labels = custom_atlas_labels
            if (atlas_path or label_path) and atlas_name:
                raise ValueError(
                    "Please specify either atlas_path and label_path or atlas_name. Atlas and label paths are only used for loading custom atlases."
                )

            if atlas_path and label_path:
                self.atlas_path = atlas_path
                self.label_path = label_path
                self.atlas_volume, self.atlas_labels = load_custom_atlas(
                    atlas_path, label_path
                )
            else:
                self._check_atlas_name()
                self.atlas_volume, self.atlas_labels = load_atlas_data(
                    atlas_name=atlas_name
                )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading settings file: {e}")
        except Exception as e:
            raise ValueError(f"Initialization error: {e}")

    def _check_atlas_name(self):
        if not self.atlas_name:
            raise ValueError(
                "When atlas_path and label_path are not specified, atlas_name must be specified."
            )

    def get_coordinates(self, non_linear=True, object_cutoff=0, use_flat=False):
        """
        Extracts pixel coordinates from the segmentation data.

        Parameters
        ----------
        non_linear : bool, optional
            Whether to use non-linear transformation from the VisuAlign markers (default is True).
        object_cutoff : int, optional
            The minimum size of objects to be considered (default is 0).
        use_flat : bool, optional
            Whether to use flat file atlas maps exported from QuickNII or VisuAlign. This is usually not needed since we can calculate them automatically. This setting is for testing and compatibility purposes (default is False).

        Returns
        -------
        None
        """
        try:
            (
                self.pixel_points,
                self.centroids,
                self.points_labels,
                self.centroids_labels,
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
            if self.custom_regions_dict is not None:
                self.points_custom_labels = map_to_custom_regions(self.custom_regions_dict, self.points_labels)
                self.centroids_custom_labels = map_to_custom_regions(self.custom_regions_dict, self.centroids_labels)

        except Exception as e:
            raise ValueError(f"Error extracting coordinates: {e}")



        except Exception as e:
            raise ValueError(f"Error extracting coordinates: {e}")


    def quantify_coordinates(self):
        """
        Quantifies the pixel coordinates by region.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If get_coordinates has not been run before running quantify_coordinates.
        """
        if not hasattr(self, "pixel_points") and not hasattr(self, "centroids"):
            raise ValueError(
                "Please run get_coordinates before running quantify_coordinates."
            )
        try:
            (self.label_df, self.per_section_df) = quantify_labeled_points(
                self.points_len,
                self.centroids_len,
                self.region_areas_list,
                self.points_labels,
                self.centroids_labels,
                self.atlas_labels,
            )
            if self.custom_regions_dict is not None:
                self.custom_label_df, self.label_df = apply_custom_regions(self.label_df, self.custom_regions_dict)
                self.custom_per_section_df = []
                for i in self.per_section_df:
                    c, i = apply_custom_regions(i, self.custom_regions_dict)
                    self.custom_per_section_df.append(c)
                self.custom_label_df
        except Exception as e:
            raise ValueError(f"Error quantifying coordinates: {e}")

    def save_analysis(self, output_folder):
        """
        Saves the pixel coordinates and pixel counts to different files in the specified output folder.

        Parameters
        ----------
        output_folder : str
            The folder where the analysis output will be saved.

        Returns
        -------
        None
        """
        try:
            save_analysis_output(
                self.pixel_points,
                self.centroids,
                self.label_df,
                self.per_section_df,
                self.points_labels,
                self.centroids_labels,
                self.points_len,
                self.centroids_len,
                self.segmentation_filenames,
                self.atlas_labels,
                output_folder,
                segmentation_folder=self.segmentation_folder,
                alignment_json=self.alignment_json,
                colour=self.colour,
                atlas_name=getattr(self, "atlas_name", None),
                custom_region_path=getattr(self, "custom_region_path", None),
                atlas_path=getattr(self, "atlas_path", None),
                label_path=getattr(self, "label_path", None),
                settings_file=getattr(self, "settings_file", None),
                prepend="",
            )
            if self.custom_regions_dict is not None:
                save_analysis_output(
                    self.pixel_points,
                    self.centroids,
                    self.custom_label_df,
                    self.per_section_df,
                    self.points_custom_labels,
                    self.centroids_custom_labels,
                    self.points_len,
                    self.centroids_len,
                    self.segmentation_filenames,
                    self.custom_atlas_labels,
                    output_folder,
                    segmentation_folder=self.segmentation_folder,
                    alignment_json=self.alignment_json,
                    colour=self.colour,
                    atlas_name=getattr(self, "atlas_name", None),
                    custom_region_path=getattr(self, "custom_region_path", None),
                    atlas_path=getattr(self, "atlas_path", None),
                    label_path=getattr(self, "label_path", None),
                    settings_file=getattr(self, "settings_file", None),
                    prepend="custom_",
                )
            print(f"Saved output to {output_folder}")
        except Exception as e:
            raise ValueError(f"Error saving analysis: {e}")

import json
import logging
import sys
from .io.atlas_loader import load_atlas_data, load_custom_atlas
from .processing.data_analysis import (
    quantify_labeled_points,
    map_to_custom_regions,
    apply_custom_regions,
)
from .io.file_operations import save_analysis_output
from .io.read_and_write import open_custom_region_file
from .processing.coordinate_extraction import folder_to_atlas_space


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PyNutil:
    """
    A class to perform brain-wide quantification and spatial analysis of serial section images.

    Methods
    -------
    __init__(...)
        Initialize the PyNutil class with segmentation, alignment, atlas and region settings.
    get_coordinates(...)
        Extract and transform pixel coordinates from segmentation files.
    quantify_coordinates()
        Quantify pixel and centroid counts by atlas regions.
    save_analysis(output_folder)
        Save the analysis output to the specified directory.
    """

    def __init__(
        self,
        segmentation_folder=None,
        alignment_json=None,
        colour=None,
        atlas_name=None,
        atlas_path=None,
        label_path=None,
        hemi_path=None,
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
        if not logger.handlers:
            file_handler = logging.FileHandler("nutil.log")
            file_handler.setLevel(logging.DEBUG)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            logger.propagate = False

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
                        if "hemi_path" in settings:
                            hemi_path = settings["hemisphere_path"]
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
                self.hemi_path = hemi_path
                self.atlas_volume, self.hemi_map, self.atlas_labels = load_custom_atlas(
                    atlas_path, hemi_path, label_path
                )
            else:
                self._check_atlas_name()
                self.atlas_volume, self.hemi_map, self.atlas_labels = load_atlas_data(
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

    def get_coordinates(
        self, non_linear=True, object_cutoff=0, use_flat=False, apply_damage_mask=True
    ):
        """
        Retrieves pixel and centroid coordinates from segmentation data,
        applies atlas-space transformations, and optionally uses a damage
        mask if specified.

        Args:
            non_linear (bool, optional): Enable non-linear transformation.
            object_cutoff (int, optional): Minimum object size.
            use_flat (bool, optional): Use flat maps if True.
            apply_damage_mask (bool, optional): Apply damage mask if True.

        Returns:
            None: Results are stored in class attributes.
        """
        try:
            (
                self.pixel_points,
                self.centroids,
                self.points_labels,
                self.centroids_labels,
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.region_areas_list,
                self.points_len,
                self.centroids_len,
                self.segmentation_filenames,
                self.per_point_undamaged,
                self.per_centroid_undamaged,
            ) = folder_to_atlas_space(
                self.segmentation_folder,
                self.alignment_json,
                self.atlas_labels,
                self.colour,
                non_linear,
                object_cutoff,
                self.atlas_volume,
                self.hemi_map,
                use_flat,
                apply_damage_mask,
            )
            self.apply_damage_mask = apply_damage_mask
            if self.custom_regions_dict is not None:
                self.points_custom_labels = map_to_custom_regions(
                    self.custom_regions_dict, self.points_labels
                )
                self.centroids_custom_labels = map_to_custom_regions(
                    self.custom_regions_dict, self.centroids_labels
                )

        except Exception as e:
            raise ValueError(f"Error extracting coordinates: {e}")

        except Exception as e:
            raise ValueError(f"Error extracting coordinates: {e}")

    def quantify_coordinates(self):
        """
        Quantifies and summarizes pixel and centroid coordinates by atlas region,
        storing the aggregated results in class attributes.

        Attributes:
            label_df (pd.DataFrame): Contains aggregated label information.
            per_section_df (list of pd.DataFrame): DataFrames with section-wise statistics.
            custom_label_df (pd.DataFrame): Label data enriched with custom regions if custom regions is set.
            custom_per_section_df (list of pd.DataFrame): Section-wise stats for custom regions if custom regions is set.

        Raises:
            ValueError: If required attributes are missing or computation fails.

        Returns:
            None
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
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.per_point_undamaged,
                self.per_centroid_undamaged,
                self.apply_damage_mask,
            )
            if self.custom_regions_dict is not None:
                self.custom_label_df, self.label_df = apply_custom_regions(
                    self.label_df, self.custom_regions_dict
                )
                self.custom_per_section_df = []
                for i in self.per_section_df:
                    c, i = apply_custom_regions(i, self.custom_regions_dict)
                    self.custom_per_section_df.append(c)
        except Exception as e:
            raise ValueError(f"Error quantifying coordinates: {e}")

    def save_analysis(self, output_folder, create_visualisations=True):
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
                self.points_hemi_labels,
                self.centroids_hemi_labels,
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
                    self.points_hemi_labels,
                    self.centroids_hemi_labels,
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

            # FIX: Remap IDs back to original if they were compressed
            if hasattr(self, "label_df") and ("original_idx" in self.label_df.columns):
                self.label_df["idx"] = self.label_df["original_idx"]
                self.label_df = self.label_df.drop(columns=["original_idx"])
                # Save CSV again with correct IDs
                self.label_df.to_csv(
                    f"{output_folder}/whole_series_report/counts.csv",
                    sep=";",
                    index=False,
                )
            elif hasattr(self, "atlas_labels") and (
                "original_idx" in self.atlas_labels.columns
            ):
                # Back-compat for older remap implementation storing original_idx on atlas_labels
                mapping = self.atlas_labels[["idx", "original_idx"]].dropna()
                mapping["idx"] = mapping["idx"].astype(int)
                mapping["original_idx"] = mapping["original_idx"].astype(int)
                if hasattr(self, "label_df") and ("idx" in self.label_df.columns):
                    self.label_df = self.label_df.merge(mapping, on="idx", how="left")
                    if "original_idx" in self.label_df.columns:
                        self.label_df["idx"] = (
                            self.label_df["original_idx"]
                            .fillna(self.label_df["idx"])
                            .astype(int)
                        )
                        self.label_df = self.label_df.drop(columns=["original_idx"])
                        self.label_df.to_csv(
                            f"{output_folder}/whole_series_report/counts.csv",
                            sep=";",
                            index=False,
                        )

            # visualisation
            if create_visualisations and self.alignment_json:
                try:
                    from .io.section_visualisation import create_section_visualisations
                    from .io.read_and_write import load_quint_json

                    alignment_data = load_quint_json(self.alignment_json)

                    logger.info("Creating section visualisations...")
                    create_section_visualisations(
                        self.segmentation_folder,
                        alignment_data,
                        self.atlas_volume,
                        self.atlas_labels,
                        output_folder,
                    )
                except Exception as e:
                    logger.error(f"visualisation failed: {e}")

            logger.info(f"Saved output to {output_folder}")
        except Exception as e:
            raise ValueError(f"Error saving analysis: {e}")

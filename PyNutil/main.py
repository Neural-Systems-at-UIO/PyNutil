import logging
import os
import re
import sys
from json import JSONDecodeError
from typing import Optional, Tuple

import numpy as np
from .io.atlas_loader import load_atlas_data, load_custom_atlas
from .processing.data_analysis import (
    quantify_labeled_points,
    quantify_intensity,
    map_to_custom_regions,
    apply_custom_regions,
)
from .io.file_operations import save_analysis_output
from .io.read_and_write import open_custom_region_file
from .processing.coordinate_extraction import (
    folder_to_atlas_space,
    folder_to_atlas_space_intensity,
)
from .io.volume_nifti import save_volume_niftis
from .processing.section_volume import (
    project_sections_to_volume as _project_sections_to_volume,
)

from .config import PyNutilConfig


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
        image_folder=None,
        alignment_json=None,
        colour=None,
        intensity_channel=None,
        atlas_name=None,
        atlas_path=None,
        label_path=None,
        hemi_path=None,
        custom_region_path=None,
        settings_file=None,
        voxel_size_um: Optional[float] = None,
        min_intensity: Optional[int] = None,
        max_intensity: Optional[int] = None,
        cellpose: bool = False,
    ):
        """
        Initializes the PyNutil class with the given parameters.

        Parameters
        ----------
        segmentation_folder : str, optional
            The folder containing the segmentation files (default is None).
        image_folder : str, optional
            The folder containing the original images for intensity quantification (default is None).
        alignment_json : str, optional
            The path to the alignment JSON file (default is None).
        colour : list, optional
            The RGB colour of the object to be quantified in the segmentation (default is None).
        intensity_channel : str, optional
            The channel to use for intensity quantification ('R', 'G', 'B', 'grayscale', 'auto').
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
        voxel_size_um : float, optional
            Only relevant when using a custom atlas. The voxel size of the atlas in micrometers (default is None).
        min_intensity : int, optional
            Only when specifying images with intensity to quantify. The minimum intensity value to include in quantification and MeshView (default is None).
        max_intensity : int, optional
            Only when specifying images with intensity to quantify. The maximum intensity value to include in quantification and MeshView (default is None).
        cellpose : bool, optional
            If True, the segmentation files are assumed to be Cellpose output (default is False).

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
                cfg = PyNutilConfig.from_settings_file(settings_file)
            else:
                cfg = PyNutilConfig(
                    segmentation_folder=segmentation_folder,
                    image_folder=image_folder,
                    alignment_json=alignment_json,
                    colour=colour,
                    intensity_channel=intensity_channel,
                    atlas_name=atlas_name,
                    atlas_path=atlas_path,
                    label_path=label_path,
                    hemi_path=hemi_path,
                    custom_region_path=custom_region_path,
                    voxel_size_um=voxel_size_um,
                    min_intensity=min_intensity,
                    max_intensity=max_intensity,
                    cellpose=cellpose,
                )

            cfg.normalize(logger=logger)
            cfg.validate()

            self.segmentation_folder = cfg.segmentation_folder
            self.image_folder = cfg.image_folder
            self.alignment_json = cfg.alignment_json
            self.colour = cfg.colour
            self.intensity_channel = cfg.intensity_channel
            self.atlas_name = cfg.atlas_name
            self.voxel_size_um = float(cfg.voxel_size_um) if cfg.voxel_size_um is not None else None
            self.min_intensity = cfg.min_intensity
            self.max_intensity = cfg.max_intensity
            self.cellpose = cfg.cellpose
            self.custom_region_path = cfg.custom_region_path

            if cfg.custom_region_path:
                custom_regions_dict, custom_atlas_labels = open_custom_region_file(
                    cfg.custom_region_path
                )
            else:
                custom_regions_dict = None
                custom_atlas_labels = None
            self.custom_regions_dict = custom_regions_dict
            self.custom_atlas_labels = custom_atlas_labels

            if cfg.atlas_path and cfg.label_path:
                self.atlas_path = cfg.atlas_path
                self.label_path = cfg.label_path
                self.hemi_path = cfg.hemi_path
                self.atlas_volume, self.hemi_map, self.atlas_labels = load_custom_atlas(
                    cfg.atlas_path, cfg.hemi_path, cfg.label_path
                )
            else:
                self._check_atlas_name()
                self.atlas_volume, self.hemi_map, self.atlas_labels = load_atlas_data(
                    atlas_name=cfg.atlas_name
                )

            # If not provided, try to infer voxel size from brainglobe atlas name
            # e.g. "allen_mouse_25um" -> 25 microns.
            if self.voxel_size_um is None and isinstance(self.atlas_name, str):
                m = re.search(r"(\d+(?:\.\d+)?)um$", self.atlas_name)
                if m:
                    try:
                        self.voxel_size_um = float(m.group(1))
                    except Exception:
                        self.voxel_size_um = None

            self.point_intensities = None
        except (FileNotFoundError, JSONDecodeError) as e:
            raise ValueError(f"Error loading settings file: {e}")
        except Exception as e:
            raise ValueError(f"Initialization error: {e}")

    def _check_atlas_name(self):
        if not self.atlas_name:
            raise ValueError(
                "When atlas_path and label_path are not specified, atlas_name must be specified."
            )

    def get_coordinates(
        self,
        non_linear=True,
        object_cutoff=0,
        use_flat=False,
        apply_damage_mask=True,
    ):
        """
        Retrieves pixel and centroid coordinates from segmentation data,
        or extracts intensity from original images,
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
            if self.image_folder:
                (
                    self.region_intensities_list,
                    self.segmentation_filenames,
                    self.pixel_points,
                    self.points_labels,
                    self.points_hemi_labels,
                    self.points_len,
                    self.point_intensities,
                ) = folder_to_atlas_space_intensity(
                    self.image_folder,
                    self.alignment_json,
                    self.atlas_labels,
                    self.intensity_channel,
                    non_linear,
                    self.atlas_volume,
                    self.hemi_map,
                    use_flat,
                    apply_damage_mask,
                    min_intensity=self.min_intensity,
                    max_intensity=self.max_intensity,
                )
                # In intensity mode, we don't have separate centroids
                self.centroids = None
                self.labeled_points_centroids = None
                self.centroids_hemi_labels = None
                self.centroids_len = None
                self.apply_damage_mask = apply_damage_mask
                return

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
                cellpose=self.cellpose,
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

    def quantify_coordinates(self):
        """
        Quantifies and summarizes pixel and centroid coordinates by atlas region,
        or summarizes intensity by atlas region,
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
        if self.image_folder:
            if not hasattr(self, "region_intensities_list"):
                raise ValueError(
                    "Please run get_coordinates before running quantify_coordinates."
                )
            try:
                (self.label_df, self.per_section_df) = quantify_intensity(
                    self.region_intensities_list, self.atlas_labels
                )
                if self.custom_regions_dict is not None:
                    self.custom_label_df, self.label_df = apply_custom_regions(
                        self.label_df, self.custom_regions_dict
                    )
                    self.custom_per_section_df = []
                    for i in self.per_section_df:
                        c, i = apply_custom_regions(i, self.custom_regions_dict)
                        self.custom_per_section_df.append(c)
                return
            except Exception as e:
                raise ValueError(f"Error quantifying intensity: {e}")

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

    def interpolate_volume(
        self,
        *,
        scale: float = 1.0,
        missing_fill: float = np.nan,
        do_interpolation: bool = True,
        k: int = 5,
        batch_size: int = 200_000,
        use_atlas_mask: bool = True,
        non_linear: bool = True,
        value_mode: str = "mean",
    ):
        """Build a 3D volume by projecting full section planes into atlas space.

        Every pixel in each section contributes to the projection.
        Background pixels contribute 0 to the signal but still increment frequency,
        which allows downstream interpolation/averaging to reflect per-section
        coverage.

        The signal is taken from the segmentation folder as a binary mask for
        `self.colour` (1.0 at matching pixels, 0.0 otherwise).

        Results are stored on the instance:
            - self.interpolated_volume (float32)
            - self.frequency_volume (uint32)

        Args:
            scale: Multiply atlas-space coordinates by this factor before binning. The output
                shape is derived from `self.atlas_volume.shape` and this scale.
            missing_fill: Value for voxels with frequency 0 (after interpolation if enabled).
            do_interpolation: If True, fill/smooth within atlas mask using kNN over observed voxels.
            k: Number of nearest neighbors (default 5).
            batch_size: KDTree query batch size.
            use_atlas_mask: If True, restrict interpolation to `self.atlas_volume != 0`.
            non_linear: If True and VisuAlign markers exist, apply marker-based deformation.
            value_mode: What each voxel represents:
                - "pixel_count": number of segmented pixels per voxel
                - "mean": mean segmentation value per voxel (averaged over all sampled pixels, including zeros)
                - "object_count": number of 2D connected components contributing to each voxel

        Returns:
            (volume, frequency_volume)
        """

        if not self.segmentation_folder and not self.image_folder:
            raise ValueError(
                "Either segmentation_folder or image_folder must be specified"
            )
        if not self.alignment_json:
            raise ValueError("alignment_json is required")
        if self.segmentation_folder and self.colour is None:
            raise ValueError(
                "colour must be set to interpolate_volume when using segmentation_folder"
            )

        atlas_shape = (
            tuple(int(x) for x in self.atlas_volume.shape)
            if getattr(self, "atlas_volume", None) is not None
            else None
        )
        if atlas_shape is None:
            raise ValueError("atlas_volume is unavailable")

        folder_to_use = (
            self.segmentation_folder if self.segmentation_folder else self.image_folder
        )

        gv, fv, dv = _project_sections_to_volume(
            segmentation_folder=folder_to_use,
            alignment_json=self.alignment_json,
            colour=self.colour,
            atlas_shape=atlas_shape,
            atlas_volume=getattr(self, "atlas_volume", None),
            scale=scale,
            missing_fill=missing_fill,
            do_interpolation=do_interpolation,
            k=k,
            batch_size=batch_size,
            use_atlas_mask=use_atlas_mask,
            non_linear=non_linear,
            value_mode=value_mode,
            intensity_channel=self.intensity_channel,
            min_intensity=self.min_intensity,
            max_intensity=self.max_intensity,
        )

        self.interpolated_volume = gv
        self.frequency_volume = fv
        self.damage_volume = dv

    def save_analysis(self, output_folder, create_visualisations=True, colormap="gray"):
        """
        Saves the pixel coordinates and pixel counts to different files in the specified output folder.

        Parameters
        ----------
        output_folder : str
            The folder where the analysis output will be saved.
        create_visualisations : bool, optional
            If True, create section visualisations (default is True).
        colormap : str, optional
            Colormap to use for intensity mode MeshView (default is "gray").
            Options: "gray", "viridis", "plasma", "hot".

        Returns
        -------
        None
        """
        try:
            save_analysis_output(
                getattr(self, "pixel_points", None),
                getattr(self, "centroids", None),
                getattr(self, "label_df", None),
                getattr(self, "per_section_df", None),
                getattr(self, "points_labels", None),
                getattr(self, "centroids_labels", None),
                getattr(self, "points_hemi_labels", None),
                getattr(self, "centroids_hemi_labels", None),
                getattr(self, "points_len", None),
                getattr(self, "centroids_len", None),
                getattr(self, "segmentation_filenames", None),
                self.atlas_labels,
                output_folder,
                segmentation_folder=self.segmentation_folder,
                image_folder=getattr(self, "image_folder", None),
                alignment_json=self.alignment_json,
                colour=self.colour,
                intensity_channel=getattr(self, "intensity_channel", None),
                point_intensities=getattr(self, "point_intensities", None),
                atlas_name=getattr(self, "atlas_name", None),
                custom_region_path=getattr(self, "custom_region_path", None),
                atlas_path=getattr(self, "atlas_path", None),
                label_path=getattr(self, "label_path", None),
                settings_file=getattr(self, "settings_file", None),
                prepend="",
                colormap=colormap,
            )

            # Save interpolated/frequency volumes if they exist.
            # We write NIfTI via nibabel to avoid relying on NRRD output tooling.
            try:
                vol = getattr(self, "interpolated_volume", None)
                freq = getattr(self, "frequency_volume", None)
                damage = getattr(self, "damage_volume", None)

                save_volume_niftis(
                    output_folder=output_folder,
                    interpolated_volume=vol,
                    frequency_volume=freq,
                    damage_volume=damage,
                    atlas_volume=getattr(self, "atlas_volume", None),
                    voxel_size_um=getattr(self, "voxel_size_um", None),
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"Saving NIfTI volumes failed: {e}")

            if self.custom_regions_dict is not None:
                save_analysis_output(
                    getattr(self, "pixel_points", None),
                    getattr(self, "centroids", None),
                    getattr(self, "custom_label_df", None),
                    getattr(self, "custom_per_section_df", None),
                    getattr(self, "points_custom_labels", None),
                    getattr(self, "centroids_custom_labels", None),
                    getattr(self, "points_hemi_labels", None),
                    getattr(self, "centroids_hemi_labels", None),
                    getattr(self, "points_len", None),
                    getattr(self, "centroids_len", None),
                    getattr(self, "segmentation_filenames", None),
                    self.custom_atlas_labels,
                    output_folder,
                    segmentation_folder=self.segmentation_folder,
                    image_folder=getattr(self, "image_folder", None),
                    alignment_json=self.alignment_json,
                    colour=self.colour,
                    intensity_channel=getattr(self, "intensity_channel", None),
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
                        self.segmentation_folder or self.image_folder,
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

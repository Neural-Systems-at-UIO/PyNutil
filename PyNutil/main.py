import logging
import re
from typing import Optional

import numpy as np
from .io.atlas_loader import load_atlas_data, load_custom_atlas
from .results import PerEntityArrays
from .processing.analysis.data_analysis import (
    quantify_labeled_points,
    quantify_intensity,
    map_to_custom_regions,
    apply_custom_regions,
)
from .io.file_operations import save_analysis_output, SaveContext
from .io.loaders import open_custom_region_file
from .processing.pipeline.batch_processor import (
    folder_to_atlas_space,
    folder_to_atlas_space_intensity,
    file_to_atlas_space_coordinates,
)
from .io.volume_nifti import save_volume_niftis
from .processing.section_volume import (
    project_sections_to_volume as _project_sections_to_volume,
)

from .config import PyNutilConfig
from .logging_utils import configure_logging


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
        coordinate_file=None,
        alignment_json=None,
        colour=None,
        intensity_channel=None,
        atlas_name=None,
        atlas_path=None,
        label_path=None,
        hemi_path=None,
        custom_region_path=None,
        voxel_size_um: Optional[float] = None,
        min_intensity: Optional[int] = None,
        max_intensity: Optional[int] = None,
        segmentation_format: str = "binary",
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
        voxel_size_um : float, optional
            Only relevant when using a custom atlas. The voxel size of the atlas in micrometers (default is None).
        min_intensity : int, optional
            Only when specifying images with intensity to quantify. The minimum intensity value to include in quantification and MeshView (default is None).
        max_intensity : int, optional
            Only when specifying images with intensity to quantify. The maximum intensity value to include in quantification and MeshView (default is None).
        segmentation_format : str, optional
            The segmentation format: "binary" for ilastik-style masks, "cellpose" for Cellpose output (default is "binary").

        Raises
        ------
        ValueError
            If both atlas_path and atlas_name are specified or if neither is specified.
        """
        # Configure logging using centralized utility (only configures if not already done)
        configure_logging()

        cfg = PyNutilConfig(
            segmentation_folder=segmentation_folder,
            image_folder=image_folder,
            coordinate_file=coordinate_file,
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
            segmentation_format=segmentation_format,
        )

        cfg.normalize(logger=logger)
        cfg.validate()

        # Store the config as the single source of truth for pipeline settings.
        # voxel_size_um may be overwritten by _infer_voxel_size below.
        self._cfg = cfg

        if cfg.custom_region_path:
            self._custom_regions_dict, self._custom_atlas_labels = (
                open_custom_region_file(cfg.custom_region_path)
            )
        else:
            self._custom_regions_dict = None
            self._custom_atlas_labels = None

        self._load_atlas_data()
        self._infer_voxel_size()

        # Extraction result (populated by get_coordinates)
        self._result = None
        # Quantification results (populated by quantify_coordinates)
        self.label_df = None
        self.per_section_df = None
        self.custom_label_df = None
        self.custom_per_section_df = None

    # ── Public config properties (tests and user code access these) ──────────

    @property
    def segmentation_folder(self):
        return self._cfg.segmentation_folder

    @property
    def image_folder(self):
        return self._cfg.image_folder

    @property
    def alignment_json(self):
        return self._cfg.alignment_json

    @property
    def colour(self):
        return self._cfg.colour

    @property
    def intensity_channel(self):
        return self._cfg.intensity_channel

    @property
    def atlas_name(self):
        return self._cfg.atlas_name

    @property
    def segmentation_format(self):
        return self._cfg.segmentation_format

    @property
    def voxel_size_um(self):
        return self._cfg.voxel_size_um

    @voxel_size_um.setter
    def voxel_size_um(self, value):
        self._cfg.voxel_size_um = value

    @property
    def custom_regions_dict(self):
        return self._custom_regions_dict

    # ── Public result properties ─────────────────────────────────────────────

    @property
    def segmentation_filenames(self):
        return self._result.segmentation_filenames if self._result else None

    # ── Atlas loading ────────────────────────────────────────────────────────

    def _load_atlas_data(self):
        """Load atlas volume, hemisphere map, and labels from config."""
        cfg = self._cfg
        if cfg.atlas_path and cfg.label_path:
            self.atlas_volume, self.hemi_map, self.atlas_labels = load_custom_atlas(
                cfg.atlas_path, cfg.hemi_path, cfg.label_path
            )
        else:
            if not cfg.atlas_name:
                raise ValueError(
                    "When atlas_path and label_path are not specified, atlas_name must be specified."
                )
            self.atlas_volume, self.hemi_map, self.atlas_labels = load_atlas_data(
                atlas_name=cfg.atlas_name
            )

    def _infer_voxel_size(self):
        """Try to infer voxel size from brainglobe atlas name."""
        if self._cfg.voxel_size_um is not None or not isinstance(self._cfg.atlas_name, str):
            return
        m = re.search(r"(\d+(?:\.\d+)?)um$", self._cfg.atlas_name)
        if m:
            try:
                self._cfg.voxel_size_um = float(m.group(1))
            except Exception:
                pass

    def get_coordinates(
        self,
        non_linear=True,
        object_cutoff=0,
        use_flat=False,
        apply_damage_mask=True,
        flat_label_path=None,
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
            flat_label_path (str, optional): Path to flatmap region-id lookup
                file (.csv or .label) used when flat files store indexed labels.

        Returns:
            None: Results are stored in class attributes.
        """
        cfg = self._cfg
        if cfg.coordinate_file:
            result = file_to_atlas_space_coordinates(
                cfg.coordinate_file,
                cfg.alignment_json,
                self.atlas_labels,
                non_linear,
                self.atlas_volume,
                self.hemi_map,
                apply_damage_mask,
            )
            map_custom_regions = True
        elif cfg.image_folder:
            result = folder_to_atlas_space_intensity(
                cfg.image_folder,
                cfg.alignment_json,
                self.atlas_labels,
                cfg.intensity_channel,
                non_linear,
                self.atlas_volume,
                self.hemi_map,
                use_flat,
                apply_damage_mask,
                flat_label_path=flat_label_path,
                min_intensity=cfg.min_intensity,
                max_intensity=cfg.max_intensity,
            )
            map_custom_regions = False
        else:
            result = folder_to_atlas_space(
                cfg.segmentation_folder,
                cfg.alignment_json,
                self.atlas_labels,
                cfg.colour,
                non_linear,
                object_cutoff,
                self.atlas_volume,
                self.hemi_map,
                use_flat,
                apply_damage_mask,
                flat_label_path=flat_label_path,
                segmentation_format=cfg.segmentation_format,
            )
            map_custom_regions = True

        self._result = result
        self._apply_damage_mask = apply_damage_mask
        if map_custom_regions and self._custom_regions_dict is not None:
            self._result.points_custom_labels = map_to_custom_regions(
                self._custom_regions_dict, result.points_labels
            )
            self._result.centroids_custom_labels = map_to_custom_regions(
                self._custom_regions_dict, result.centroids_labels
            )

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
        r = self._result
        if self._cfg.image_folder:
            if r is None or not hasattr(r, "region_intensities_list"):
                raise ValueError(
                    "Please run get_coordinates before running quantify_coordinates."
                )
            self.label_df, self.per_section_df = quantify_intensity(
                r.region_intensities_list, self.atlas_labels
            )
        else:
            if r is None:
                raise ValueError(
                    "Please run get_coordinates before running quantify_coordinates."
                )
            self.label_df, self.per_section_df = quantify_labeled_points(
                PerEntityArrays(
                    labels=r.points_labels,
                    hemi_labels=r.points_hemi_labels,
                    undamaged=r.per_point_undamaged,
                    section_lengths=r.total_points_len,
                ),
                PerEntityArrays(
                    labels=r.centroids_labels,
                    hemi_labels=r.centroids_hemi_labels,
                    undamaged=r.per_centroid_undamaged,
                    section_lengths=r.total_centroids_len,
                ),
                r.region_areas_list,
                self.atlas_labels,
                self._apply_damage_mask,
            )
        if self._custom_regions_dict is not None:
            self.custom_label_df, self.label_df = apply_custom_regions(
                self.label_df, self._custom_regions_dict
            )
            self.custom_per_section_df = []
            for section_df in self.per_section_df:
                custom_df, section_df = apply_custom_regions(
                    section_df, self._custom_regions_dict
                )
                self.custom_per_section_df.append(custom_df)

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
        cfg = self._cfg
        if not cfg.segmentation_folder and not cfg.image_folder:
            raise ValueError(
                "Either segmentation_folder or image_folder must be specified"
            )
        if not cfg.alignment_json:
            raise ValueError("alignment_json is required")
        if (
            cfg.segmentation_folder
            and cfg.segmentation_format != "cellpose"
            and cfg.colour is None
        ):
            raise ValueError(
                "colour must be set to interpolate_volume when using segmentation_folder"
            )

        if self.atlas_volume is None:
            raise ValueError("atlas_volume is unavailable")
        atlas_shape = tuple(int(x) for x in self.atlas_volume.shape)

        folder_to_use = cfg.segmentation_folder or cfg.image_folder

        gv, fv, dv = _project_sections_to_volume(
            segmentation_folder=folder_to_use,
            alignment_json=cfg.alignment_json,
            colour=cfg.colour,
            atlas_shape=atlas_shape,
            atlas_volume=self.atlas_volume,
            scale=scale,
            missing_fill=missing_fill,
            do_interpolation=do_interpolation,
            k=k,
            batch_size=batch_size,
            use_atlas_mask=use_atlas_mask,
            non_linear=non_linear,
            value_mode=value_mode,
            segmentation_format=cfg.segmentation_format,
            segmentation_mode=bool(cfg.segmentation_folder),
            intensity_channel=cfg.intensity_channel,
            min_intensity=cfg.min_intensity,
            max_intensity=cfg.max_intensity,
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
        base_ctx = self._build_save_context(colormap)

        self._save_analysis_variant(
            base_ctx,
            output_folder,
            label_df=self.label_df,
            per_section_df=self.per_section_df,
            points_labels=self._result.points_labels if self._result else None,
            centroids_labels=self._result.centroids_labels if self._result else None,
            atlas_labels=self.atlas_labels,
            prepend="",
        )
        self._save_volumes(output_folder)

        if self._custom_regions_dict is not None:
            self._save_analysis_variant(
                base_ctx,
                output_folder,
                label_df=self.custom_label_df,
                per_section_df=self.custom_per_section_df,
                points_labels=self._result.points_custom_labels if self._result else None,
                centroids_labels=self._result.centroids_custom_labels if self._result else None,
                atlas_labels=self._custom_atlas_labels,
                prepend="custom_",
                colormap="gray",
            )

        self._remap_compressed_ids(output_folder)

        if create_visualisations and self._cfg.alignment_json:
            self._create_visualisations(output_folder)

        logger.info(f"Saved output to {output_folder}")

    def _filter_undamaged(self, full_arr, undamaged_mask):
        """Filter *full_arr* to undamaged-only using *undamaged_mask* if lengths match."""
        if (
            undamaged_mask is not None
            and full_arr is not None
            and len(undamaged_mask) == len(full_arr)
        ):
            return full_arr[undamaged_mask]
        return full_arr

    def _build_save_context(self, colormap):
        """Build a SaveContext from the stored config and extraction result."""
        r = self._result
        und_p = r.per_point_undamaged if r else None
        und_c = r.per_centroid_undamaged if r else None
        return SaveContext(
            pixel_points=r.pixel_points if r else None,
            centroids=r.centroids if r else None,
            points_hemi_labels=self._filter_undamaged(
                r.points_hemi_labels if r else None, und_p
            ),
            centroids_hemi_labels=self._filter_undamaged(
                r.centroids_hemi_labels if r else None, und_c
            ),
            points_len=r.points_len if r else None,
            centroids_len=r.centroids_len if r else None,
            segmentation_filenames=r.segmentation_filenames if r else None,
            point_intensities=r.point_intensities if r else None,
            config=self._cfg,
            colormap=colormap,
        )

    def _save_analysis_variant(
        self,
        base_ctx,
        output_folder,
        *,
        label_df,
        per_section_df,
        points_labels,
        centroids_labels,
        atlas_labels,
        prepend,
        colormap=None,
    ):
        """Save one analysis variant (primary or custom-region) to output files."""
        r = self._result
        und_p = r.per_point_undamaged if r else None
        und_c = r.per_centroid_undamaged if r else None
        base_ctx.label_df = label_df
        base_ctx.per_section_df = per_section_df
        base_ctx.labeled_points = self._filter_undamaged(points_labels, und_p)
        base_ctx.labeled_points_centroids = self._filter_undamaged(centroids_labels, und_c)
        base_ctx.atlas_labels = atlas_labels
        base_ctx.prepend = prepend
        if colormap is not None:
            base_ctx.colormap = colormap
        save_analysis_output(base_ctx, output_folder)

    def _save_volumes(self, output_folder):
        """Save interpolated / frequency / damage NIfTI volumes if they exist."""
        try:
            save_volume_niftis(
                output_folder=output_folder,
                interpolated_volume=getattr(self, "interpolated_volume", None),
                frequency_volume=getattr(self, "frequency_volume", None),
                damage_volume=getattr(self, "damage_volume", None),
                atlas_volume=self.atlas_volume,
                voxel_size_um=self._cfg.voxel_size_um,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"Saving NIfTI volumes failed: {e}")

    def _remap_compressed_ids(self, output_folder):
        """Restore original atlas IDs if they were compressed during loading."""
        if self.label_df is not None and "original_idx" in self.label_df.columns:
            self.label_df["idx"] = self.label_df["original_idx"]
            self.label_df = self.label_df.drop(columns=["original_idx"])
            self.label_df.to_csv(
                f"{output_folder}/whole_series_report/counts.csv",
                sep=";",
                index=False,
            )

    def _create_visualisations(self, output_folder):
        """Create section visualisation PNGs."""
        try:
            from .io.section_visualisation import create_section_visualisations
            from .processing.adapters.registry import load_registration
            from .processing.adapters.segmentation import SegmentationAdapterRegistry

            reg_data = load_registration(
                self._cfg.alignment_json, apply_deformation=False, apply_damage=False
            )
            adapter = SegmentationAdapterRegistry.get(self._cfg.segmentation_format)

            logger.info("Creating section visualisations...")
            create_section_visualisations(
                self._cfg.segmentation_folder or self._cfg.image_folder,
                reg_data.slices,
                self.atlas_volume,
                self.atlas_labels,
                output_folder,
                adapter=adapter,
                pixel_id=self._cfg.colour,
            )
        except Exception as e:
            logger.error(f"visualisation failed: {e}")

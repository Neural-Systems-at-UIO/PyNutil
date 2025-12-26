import json
import logging
import os
import re
import sys
from typing import Optional, Tuple

import numpy as np
import cv2
from .io.atlas_loader import load_atlas_data, load_custom_atlas
from .io.read_and_write import load_quint_json
from .processing.data_analysis import (
    quantify_labeled_points,
    map_to_custom_regions,
    apply_custom_regions,
)
from .io.file_operations import save_analysis_output
from .io.read_and_write import open_custom_region_file
from .processing.coordinate_extraction import folder_to_atlas_space
from .processing.transformations import (
    transform_to_registration,
    transform_to_atlas_space,
)
from .processing.utils import number_sections
from .processing.visualign_deformations import triangulate, transform_vec


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
        voxel_size_um: Optional[float] = None,
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
                    if "voxel_size_um" in settings:
                        voxel_size_um = settings["voxel_size_um"]
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
            self.voxel_size_um = float(voxel_size_um) if voxel_size_um is not None else None
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

            # If not provided, try to infer voxel size from brainglobe atlas name
            # e.g. "allen_mouse_25um" -> 25 microns.
            if self.voxel_size_um is None and isinstance(self.atlas_name, str):
                m = re.search(r"(\d+(?:\.\d+)?)um$", self.atlas_name)
                if m:
                    try:
                        self.voxel_size_um = float(m.group(1))
                    except Exception:
                        self.voxel_size_um = None
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

    def interpolate_volume(
        self,
        *,
        source: str = "pixels",
        scale: float = 1.0,
        shape: Optional[Tuple[int, int, int]] = None,
        return_frequencies: bool = False,
        missing_fill: float = np.nan,
        do_interpolation: bool = False,
        k: int = 1,
        weights: str = "uniform",
        batch_size: int = 500_000,
        use_atlas_mask: bool = True,
    ):
        """Create a 3D voxel volume from atlas-space coordinates.

        This method takes the atlas-space coordinates produced by `get_coordinates`
        (`self.pixel_points` or `self.centroids`) and bins them into a 3D grid.
        Optionally, it fills voxels with no data using a nearest-neighbor
        interpolation inside the atlas mask.

        Notes:
            - `do_interpolation=True` requires SciPy (uses `scipy.spatial.cKDTree`).
              SciPy is treated as an optional dependency.
            - This can be memory/time intensive at full atlas resolution.
              Consider using fewer points (e.g. centroids) or a smaller `shape`.

        Args:
            source: "pixels" or "centroids".
            shape: Output volume shape (x, y, z). Defaults to `self.atlas_volume.shape`.
            scale: Multiply coordinates by this factor before voxelization.
            return_frequencies: If True, also return a frequency volume (voxel hit counts).
            missing_fill: Value to place in voxels with no hits when `do_interpolation=False`.
            do_interpolation: If True, fill missing voxels inside atlas mask by nearest-neighbor.
            k: Number of nearest neighbors (k>=1). k>1 averages neighbors.
            weights: "uniform" or "distance" (only used when k>1).
            batch_size: Query batch size for interpolation to avoid huge allocations.
            use_atlas_mask: If True, interpolate only within `self.atlas_volume != 0`.

        Returns:
            volume or (volume, frequency_volume)
        """
        if not hasattr(self, "pixel_points") and not hasattr(self, "centroids"):
            raise ValueError("Please run get_coordinates before interpolate_volume.")

        if source not in {"pixels", "centroids"}:
            raise ValueError("source must be 'pixels' or 'centroids'")

        points_list = self.pixel_points if source == "pixels" else self.centroids
        if points_list is None:
            raise ValueError(f"No point data found for source={source!r}")

        if not hasattr(self, "atlas_volume") or self.atlas_volume is None:
            if shape is None:
                raise ValueError("shape must be provided when atlas_volume is unavailable")
        else:
            atlas_shape = tuple(int(x) for x in self.atlas_volume.shape)
            if shape is None:
                # Shape is derived from atlas_shape and scale.
                if scale <= 0:
                    raise ValueError("scale must be > 0")
                shape = tuple(max(1, int(round(s * float(scale)))) for s in atlas_shape)
            else:
                # Back-compat escape hatch: allow explicit shape, but don't also scale.
                if scale != 1.0:
                    raise ValueError(
                        "Do not pass both shape and scale; shape is derived from scale and atlas shape."
                    )

        # Stack points across sections.
        valid_arrays = [p for p in points_list if isinstance(p, np.ndarray) and p.size]
        if not valid_arrays:
            gv = np.full(shape, missing_fill, dtype=np.float32)
            fv = np.zeros(shape, dtype=np.uint32)
            return (gv, fv) if return_frequencies else gv

        pts = np.vstack(valid_arrays).astype(np.float32, copy=False)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"Expected Nx3 atlas-space coordinates; got array of shape {pts.shape}"
            )

        # For derived-shape usage, scale also defines coordinate scaling.
        if scale != 1.0:
            pts = pts * float(scale)

        # Convert to voxel indices in atlas (x, y, z) order.
        idx = np.rint(pts).astype(np.int64, copy=False)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
        sx, sy, sz = shape
        in_bounds = (x >= 0) & (x < sx) & (y >= 0) & (y < sy) & (z >= 0) & (z < sz)
        if not np.any(in_bounds):
            gv = np.full(shape, missing_fill, dtype=np.float32)
            fv = np.zeros(shape, dtype=np.uint32)
            return (gv, fv) if return_frequencies else gv

        x = x[in_bounds]
        y = y[in_bounds]
        z = z[in_bounds]

        gv = np.zeros(shape, dtype=np.float32)
        fv = np.zeros(shape, dtype=np.uint32)

        # Count hits per voxel (sum and frequency are the same for binary points).
        np.add.at(gv, (x, y, z), 1.0)
        np.add.at(fv, (x, y, z), 1)

        if do_interpolation:
            try:
                from scipy.spatial import cKDTree  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "SciPy is required for do_interpolation=True (pip install scipy)."
                ) from exc

            if k < 1:
                raise ValueError("k must be >= 1")
            if k > 1 and weights not in {"uniform", "distance"}:
                raise ValueError("weights must be 'uniform' or 'distance'")

            if use_atlas_mask and hasattr(self, "atlas_volume") and self.atlas_volume is not None:
                atlas_mask = self.atlas_volume != 0
                if atlas_mask.shape != gv.shape:
                    atlas_mask = None
            else:
                atlas_mask = None

            # Fit points are voxels where we observed at least one hit.
            fit_mask = fv != 0
            if atlas_mask is not None:
                # Only interpolate within atlas mask, and only fit from points within the mask.
                target_mask = atlas_mask
                fit_mask &= atlas_mask
            else:
                # Interpolate over the entire volume.
                target_mask = np.ones_like(fv, dtype=bool)

            # If we're using the ccfv3augmented reference pipeline, use its interpolation.
            # This matches the original helper logic (mask built from a resampled atlas).
            use_reference = isinstance(getattr(self, "atlas_name", None), str) and (
                "ccfv3augmented" in str(getattr(self, "atlas_name", "")).lower()
            )
            if use_reference:
                from .processing.reference_volume_port import interpolate as _ref_interpolate

                base_voxel_um = (
                    float(self.voxel_size_um)
                    if getattr(self, "voxel_size_um", None) is not None
                    else 25.0
                )
                # If we scaled coordinates/shape by `scale`, the effective voxel size increases.
                resolution_um = float(base_voxel_um / float(scale)) if float(scale) != 0 else base_voxel_um
                gv = _ref_interpolate(gv, fv, k=k, resolution=resolution_um)
            else:
                if np.any(target_mask) and np.any(fit_mask):
                    fit_pts = np.column_stack(np.nonzero(fit_mask)).astype(np.float32, copy=False)
                    fit_vals = gv[fit_mask].astype(np.float32, copy=False)
                    tree = cKDTree(fit_pts)

                    # IMPORTANT: also recompute values at fitted voxels so every voxel becomes
                    # an average of k-nearest observed voxels.
                    query_pts = np.column_stack(np.nonzero(target_mask)).astype(
                        np.float32, copy=False
                    )
                    out_vals = np.empty((query_pts.shape[0],), dtype=np.float32)
                    eps = 1e-12

                    for start in range(0, query_pts.shape[0], batch_size):
                        end = min(start + batch_size, query_pts.shape[0])
                        q = query_pts[start:end]
                        dist, ind = tree.query(q, k=k)
                        if k == 1:
                            out_vals[start:end] = fit_vals[ind]
                        else:
                            neigh_vals = fit_vals[ind]
                            if weights == "uniform":
                                out_vals[start:end] = neigh_vals.mean(axis=1)
                            else:
                                w = 1.0 / (dist * dist + eps)
                                out_vals[start:end] = (neigh_vals * w).sum(axis=1) / w.sum(axis=1)

                    gv[target_mask] = out_vals

            # If we restricted interpolation to atlas_mask, fill outside with missing_fill.
            if atlas_mask is not None and missing_fill is not None and not (missing_fill == 0):
                gv[~atlas_mask] = float(missing_fill)

            # keep fv unchanged; it represents observed hits, not interpolated values

        else:
            # If not interpolating, mark missing voxels.
            if missing_fill is not None and not (missing_fill == 0):
                gv = gv.astype(np.float32, copy=False)
                gv[fv == 0] = float(missing_fill)

        # Expose for downstream saving (e.g., save_analysis writes NRRD volumes).
        self.interpolated_volume = gv
        self.frequency_volume = fv

        return (gv, fv) if return_frequencies else gv

    def build_volume_from_sections(
        self,
        *,
        scale: float = 1.0,
        shape: Optional[Tuple[int, int, int]] = None,
        missing_fill: float = np.nan,
        do_interpolation: bool = True,
        k: int = 5,
        weights: str = "uniform",
        batch_size: int = 200_000,
        use_atlas_mask: bool = True,
        non_linear: bool = True,
    ):
        """Build a 3D volume by projecting full section planes into atlas space.

        This mirrors the key behavior of the reference implementation you shared:
        every pixel in each section plane contributes (background pixels contribute
        0 to the signal but still increment frequency), so interpolation and
        averaging behave like a per-slice plane accumulation.

        The signal is taken from the segmentation folder as a binary mask for
        `self.colour` (1.0 at matching pixels, 0.0 otherwise).

        Results are stored on the instance:
            - self.interpolated_volume (float32)
            - self.frequency_volume (uint32)

        Args:
            scale: Multiply atlas-space coordinates by this factor before binning. The output
                shape is derived from `self.atlas_volume.shape` and this scale.
            shape: Deprecated escape hatch. Do not pass together with scale.
            missing_fill: Value for voxels with frequency 0 (after interpolation if enabled).
            do_interpolation: If True, fill/smooth within atlas mask using kNN over observed voxels.
            k: Number of nearest neighbors (default 5).
            weights: "uniform" or "distance" (used when k>1).
            batch_size: KDTree query batch size.
            use_atlas_mask: If True, restrict interpolation to `self.atlas_volume != 0`.
            non_linear: If True and VisuAlign markers exist, apply marker-based deformation.

        Returns:
            (volume, frequency_volume)
        """

        if not self.segmentation_folder or not self.alignment_json:
            raise ValueError("segmentation_folder and alignment_json are required")
        if self.colour is None:
            raise ValueError("colour must be set to build_volume_from_sections")
        if not hasattr(self, "atlas_volume") or self.atlas_volume is None:
            if shape is None:
                raise ValueError("shape must be provided when atlas_volume is unavailable")
        else:
            atlas_shape = tuple(int(x) for x in self.atlas_volume.shape)
            if shape is None:
                if scale <= 0:
                    raise ValueError("scale must be > 0")
                shape = tuple(max(1, int(round(s * float(scale)))) for s in atlas_shape)
            else:
                if scale != 1.0:
                    raise ValueError(
                        "Do not pass both shape and scale; shape is derived from scale and atlas shape."
                    )

        quint_json = load_quint_json(self.alignment_json)
        slices = quint_json["slices"]

        # Build fast lookup from section nr -> slice dict
        slice_by_nr = {int(s.get("nr")): s for s in slices if s.get("nr") is not None}

        # Load segmentation paths (same logic as coordinate extraction uses for numbering)
        # We rely on filenames containing _s### so they map to alignment slices.
        seg_paths = []
        for name in os.listdir(self.segmentation_folder):
            p = os.path.join(self.segmentation_folder, name)
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"} or name.endswith(
                ".dzip"
            ):
                seg_paths.append(p)
        seg_paths.sort()

        gv = np.zeros(shape, dtype=np.float32)
        fv = np.zeros(shape, dtype=np.uint32)

        sx, sy, sz = shape
        colour = np.array(self.colour, dtype=np.uint8)

        for seg_path in seg_paths:
            seg_nr = int(number_sections([seg_path])[0])
            slice_dict = slice_by_nr.get(seg_nr)
            if not slice_dict or not slice_dict.get("anchoring"):
                continue

            seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if seg is None:
                continue
            if seg.ndim == 2:
                # If grayscale segmentation, treat non-zero as signal.
                mask = (seg != 0).astype(np.float32)
                seg_height, seg_width = seg.shape
            else:
                seg = seg[:, :, :3]
                mask = np.all(seg == colour[None, None, :], axis=2).astype(np.float32)
                seg_height, seg_width = seg.shape[:2]

            reg_height, reg_width = int(slice_dict["height"]), int(slice_dict["width"])
            y_scale, x_scale = transform_to_registration(
                seg_height, seg_width, reg_height, reg_width
            )

            # Build full pixel grid in segmentation space, scale to registration space.
            yy, xx = np.indices((seg_height, seg_width), dtype=np.float32)
            scaled_y = yy * float(y_scale)
            scaled_x = xx * float(x_scale)

            # Apply VisuAlign marker warp (registration-space deformation) if available.
            if non_linear and "markers" in slice_dict:
                tri = triangulate(reg_width, reg_height, slice_dict["markers"])
                flat_x = scaled_x.reshape(-1)
                flat_y = scaled_y.reshape(-1)
                new_x, new_y = transform_vec(tri, flat_x, flat_y)
            else:
                new_x = scaled_x.reshape(-1)
                new_y = scaled_y.reshape(-1)

            # Map to atlas space using anchoring.
            coords = transform_to_atlas_space(
                slice_dict["anchoring"], new_y, new_x, reg_height, reg_width
            )
            if scale != 1.0:
                coords = coords * float(scale)

            idx = np.rint(coords).astype(np.int64, copy=False)
            x = idx[:, 0]
            y = idx[:, 1]
            z = idx[:, 2]
            inb = (x >= 0) & (x < sx) & (y >= 0) & (y < sy) & (z >= 0) & (z < sz)
            if not np.any(inb):
                continue

            x = x[inb]
            y = y[inb]
            z = z[inb]
            vals = mask.reshape(-1)[inb]

            # Faithful behavior: frequency increments for every projected pixel,
            # signal adds mask value (0 or 1).
            np.add.at(fv, (x, y, z), 1)
            np.add.at(gv, (x, y, z), vals)

        # Fill missing before/after interpolation similarly to the reference pipeline.
        if do_interpolation:
            # Reuse the same kNN smoothing logic as interpolate_volume, but treat gv/fv
            # as an accumulated plane volume.
            if k < 1:
                raise ValueError("k must be >= 1")
            if k > 1 and weights not in {"uniform", "distance"}:
                raise ValueError("weights must be 'uniform' or 'distance'")

            try:
                from scipy.spatial import cKDTree  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "SciPy is required for do_interpolation=True (pip install scipy)."
                ) from exc

            if use_atlas_mask and getattr(self, "atlas_volume", None) is not None:
                atlas_mask = self.atlas_volume != 0
                if atlas_mask.shape != gv.shape:
                    atlas_mask = None
            else:
                atlas_mask = None

            fit_mask = fv != 0
            if atlas_mask is not None:
                target_mask = atlas_mask
                fit_mask &= atlas_mask
            else:
                target_mask = np.ones_like(fv, dtype=bool)

            use_reference = isinstance(getattr(self, "atlas_name", None), str) and (
                "ccfv3augmented" in str(getattr(self, "atlas_name", "")).lower()
            )
            if use_reference:
                from .processing.reference_volume_port import interpolate as _ref_interpolate

                base_voxel_um = (
                    float(self.voxel_size_um)
                    if getattr(self, "voxel_size_um", None) is not None
                    else 25.0
                )
                resolution_um = float(base_voxel_um / float(scale)) if float(scale) != 0 else base_voxel_um
                gv = _ref_interpolate(gv, fv, k=k, resolution=resolution_um)
            else:
                if np.any(target_mask) and np.any(fit_mask):
                    fit_pts = np.column_stack(np.nonzero(fit_mask)).astype(np.float32, copy=False)
                    fit_vals = gv[fit_mask].astype(np.float32, copy=False)
                    tree = cKDTree(fit_pts)

                    query_pts = np.column_stack(np.nonzero(target_mask)).astype(
                        np.float32, copy=False
                    )
                    out_vals = np.empty((query_pts.shape[0],), dtype=np.float32)
                    eps = 1e-12
                    for start in range(0, query_pts.shape[0], batch_size):
                        end = min(start + batch_size, query_pts.shape[0])
                        q = query_pts[start:end]
                        dist, ind = tree.query(q, k=k)
                        if k == 1:
                            out_vals[start:end] = fit_vals[ind]
                        else:
                            neigh_vals = fit_vals[ind]
                            if weights == "uniform":
                                out_vals[start:end] = neigh_vals.mean(axis=1)
                            else:
                                w = 1.0 / (dist * dist + eps)
                                out_vals[start:end] = (neigh_vals * w).sum(axis=1) / w.sum(axis=1)

                    # Reference behavior: only mask voxels are defined after interpolation;
                    # outside-mask is 0.
                    if atlas_mask is not None:
                        out = np.zeros_like(gv)
                        out[target_mask] = out_vals
                        gv = out
                    else:
                        gv[target_mask] = out_vals
        else:
            if missing_fill is not None and not (missing_fill == 0):
                gv[fv == 0] = float(missing_fill)

        self.interpolated_volume = gv.astype(np.float32, copy=False)
        self.frequency_volume = fv.astype(np.uint32, copy=False)
        return self.interpolated_volume, self.frequency_volume

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

            # Save interpolated/frequency volumes if they exist.
            # We write NIfTI via nibabel to avoid relying on NRRD output tooling.
            try:
                import nibabel as nib  # type: ignore

                vol = getattr(self, "interpolated_volume", None)
                freq = getattr(self, "frequency_volume", None)

                if vol is not None or freq is not None:
                    out_dir = f"{output_folder}/interpolated_volume"
                    os.makedirs(out_dir, exist_ok=True)

                from .processing.reference_volume_port import write_nifti as _write_nifti

                def _scale_to_uint8(data: np.ndarray) -> np.ndarray:
                    arr = np.asarray(data, dtype=np.float32)
                    finite = np.isfinite(arr)
                    if not np.any(finite):
                        return np.zeros(arr.shape, dtype=np.uint8)

                    vmin = float(np.min(arr[finite]))
                    vmax = float(np.max(arr[finite]))

                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        out = np.zeros(arr.shape, dtype=np.uint8)
                        return out

                    scaled = (arr - vmin) * (255.0 / (vmax - vmin))
                    scaled = np.clip(scaled, 0.0, 255.0)

                    out = np.zeros(arr.shape, dtype=np.uint8)
                    out[finite] = scaled[finite].round().astype(np.uint8)
                    return out

                atlas_shape = (
                    np.array(self.atlas_volume.shape, dtype=np.float32)
                    if getattr(self, "atlas_volume", None) is not None
                    else None
                )

                base_voxel_um = (
                    float(self.voxel_size_um)
                    if getattr(self, "voxel_size_um", None) is not None
                    else 1.0
                )

                def _resolution_um_for_volume(volume: np.ndarray) -> float:
                    # Always write isotropic spacing (single scalar), even when the volume
                    # shape differs from the atlas shape.
                    if atlas_shape is None:
                        return float(base_voxel_um)

                    vol_shape = np.array(volume.shape, dtype=np.float32)
                    if vol_shape.shape != (3,) or np.any(vol_shape <= 0):
                        return float(base_voxel_um)

                    implied = atlas_shape / vol_shape  # per-axis scale factors
                    iso_scale = float(np.median(implied))

                    # If the implied scale differs across axes, the world-FOV cannot be
                    # preserved in all axes with isotropic voxels. We still write isotropic
                    # spacing, but log a warning for visibility.
                    if np.max(implied) - np.min(implied) > 1e-3:
                        logger.warning(
                            "Non-uniform volume scaling detected (atlas_shape=%s, volume_shape=%s). "
                            "Writing isotropic voxel spacing using median scale %.6f.",
                            tuple(int(x) for x in atlas_shape),
                            tuple(int(x) for x in vol_shape),
                            iso_scale,
                        )

                    return float(base_voxel_um * iso_scale)

                if vol is not None:
                    _write_nifti(
                        _scale_to_uint8(np.asarray(vol)),
                        _resolution_um_for_volume(np.asarray(vol)),
                        f"{output_folder}/interpolated_volume/interpolated_volume",
                    )

                if freq is not None:
                    _write_nifti(
                        _scale_to_uint8(np.asarray(freq)),
                        _resolution_um_for_volume(np.asarray(freq)),
                        f"{output_folder}/interpolated_volume/frequency_volume",
                    )
            except Exception as e:
                logger.error(f"Saving NIfTI volumes failed: {e}")

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

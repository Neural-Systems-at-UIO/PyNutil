from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from .adapters import read_alignment
from .adapters.segmentation import SegmentationAdapterRegistry
from .atlas_map import transform_to_atlas_space
from .utils import (
    convert_to_intensity,
    discover_image_files,
    resize_mask_nearest,
)
from ..io.loaders import number_sections


@dataclass(frozen=True)
class VolumeConfig:
    """Immutable configuration for section-to-volume projection.

    Groups the per-run processing parameters so they can be passed as a
    single object to ``_process_one_section`` instead of nine positional
    arguments.
    """

    segmentation_adapter: object
    segmentation_mode: bool
    colour_arr: Optional[np.ndarray]
    intensity_channel: str
    min_intensity: Optional[int]
    max_intensity: Optional[int]
    scale: float
    non_linear: bool
    value_mode: str


@dataclass(frozen=True)
class InterpolationConfig:
    """Immutable configuration for volume interpolation and finalisation.

    Groups the post-accumulation parameters so they can be passed as a
    single object to ``_finalize_volumes``.
    """

    do_interpolation: bool
    missing_fill: float
    use_atlas_mask: bool
    atlas_volume: Optional[np.ndarray]
    k: int
    batch_size: int


def derive_shape_from_atlas(
    *,
    atlas_shape: Tuple[int, int, int],
    scale: float,
) -> Tuple[int, int, int]:
    """Derive an output shape from atlas shape + scale."""

    if scale <= 0:
        raise ValueError("scale must be > 0")

    return tuple(max(1, int(round(int(s) * float(scale)))) for s in atlas_shape)


def _knn_batch_query(tree, fit_vals, query_pts, k, batch_size, mode):
    """Query *tree* in batches and return interpolated values."""
    out_vals = np.empty((query_pts.shape[0],), dtype=np.float32)
    for start in tqdm(range(0, query_pts.shape[0], batch_size), desc="filling volume"):
        end = min(start + batch_size, query_pts.shape[0])
        _, ind = tree.query(query_pts[start:end], k=k)
        if k == 1:
            out_vals[start:end] = fit_vals[ind]
        else:
            neigh_vals = fit_vals[ind]
            out_vals[start:end] = (
                neigh_vals.mean(axis=1) if mode == "mean" else neigh_vals.max(axis=1)
            )
    return out_vals


def _knn_interpolate_generic(
    *,
    gv: np.ndarray,
    fv: np.ndarray,
    atlas_mask: Optional[np.ndarray],
    k: int,
    batch_size: int,
    mode: str = "mean",
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SciPy is required for do_interpolation=True") from exc

    if k < 1:
        raise ValueError("k must be >= 1")

    fit_mask = fv != 0
    if atlas_mask is not None:
        target_mask = atlas_mask
        fit_mask &= atlas_mask
    else:
        target_mask = np.ones_like(fv, dtype=bool)

    if not (np.any(target_mask) and np.any(fit_mask)):
        return gv

    fit_pts = np.column_stack(np.nonzero(fit_mask)).astype(np.float32, copy=False)
    fit_vals = gv[fit_mask].astype(np.float32, copy=False)
    tree = cKDTree(fit_pts)

    query_pts = np.column_stack(np.nonzero(target_mask)).astype(np.float32, copy=False)
    out_vals = _knn_batch_query(tree, fit_vals, query_pts, k, batch_size, mode)

    if atlas_mask is not None:
        out = np.zeros_like(gv)
        out[target_mask] = out_vals
        return out

    gv[target_mask] = out_vals
    return gv


def _read_section_signal(
    seg_path: str,
    vol_cfg: VolumeConfig,
):
    """Load an image and return (seg_values, mask, seg_height, seg_width) or None."""
    if vol_cfg.segmentation_mode:
        seg = vol_cfg.segmentation_adapter.load(seg_path)
    else:
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None

    if not vol_cfg.segmentation_mode:
        # Intensity mode
        seg_values = convert_to_intensity(seg, vol_cfg.intensity_channel)
        if vol_cfg.min_intensity is not None:
            seg_values[seg_values < vol_cfg.min_intensity] = 0
        if vol_cfg.max_intensity is not None:
            seg_values[seg_values > vol_cfg.max_intensity] = 0
        mask = (seg_values != 0).astype(np.float32, copy=False)
    else:
        # Segmentation mode via adapter (supports binary/cellpose/custom)
        pixel_id = vol_cfg.colour_arr.tolist() if vol_cfg.colour_arr is not None else None
        mask = vol_cfg.segmentation_adapter.create_binary_mask(seg, pixel_id=pixel_id).astype(
            np.float32, copy=False
        )
        seg_values = mask

    seg_height, seg_width = seg.shape[:2]
    return seg_values, mask, seg_height, seg_width


def _sample_and_deform_plane(
    slice_info,
    values_reg: np.ndarray,
    damage_reg: Optional[np.ndarray],
    vol_cfg: VolumeConfig,
):
    """Construct a sampling grid, optionally deform, and remap values.

    Returns (sampled_2d, vals_flat, damage_vals, flat_x, flat_y, plane_h, plane_w).
    """
    reg_height, reg_width = slice_info.height, slice_info.width
    scale = vol_cfg.scale
    anch = slice_info.anchoring
    u = np.asarray(anch[3:6], dtype=np.float32)
    v = np.asarray(anch[6:9], dtype=np.float32)
    plane_w = max(1, int(round(float(np.linalg.norm(u)) * float(scale))))
    plane_h = max(1, int(round(float(np.linalg.norm(v)) * float(scale))))

    yy, xx = np.indices((plane_h, plane_w), dtype=np.float32)
    reg_x = (xx + 0.5) * (float(reg_width) / float(plane_w))
    reg_y = (yy + 0.5) * (float(reg_height) / float(plane_h))

    flat_x = reg_x.reshape(-1)
    flat_y = reg_y.reshape(-1)

    if vol_cfg.non_linear and slice_info.forward_deformation is not None:
        new_x, new_y = slice_info.forward_deformation(flat_x, flat_y)
        map_x = new_x.reshape((plane_h, plane_w)).astype(np.float32, copy=False)
        map_y = new_y.reshape((plane_h, plane_w)).astype(np.float32, copy=False)

    else:
        map_x = reg_x.astype(np.float32, copy=False)
        map_y = reg_y.astype(np.float32, copy=False)

    sampled_2d = cv2.remap(
        values_reg,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    vals_flat = sampled_2d.reshape(-1).astype(np.float32, copy=False)

    damage_vals = None
    if damage_reg is not None:
        sampled_damage = cv2.remap(
            damage_reg,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        damage_vals = sampled_damage.reshape(-1)

    return sampled_2d, vals_flat, damage_vals, flat_x, flat_y, plane_h, plane_w


def _accumulate_object_counts(
    sampled_2d: np.ndarray,
    inb: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seg_nr: int,
    out_shape: Tuple[int, int, int],
    ov_flat: np.ndarray,
):
    """Count unique 2-D connected components per voxel, accumulating into *ov_flat*."""
    sampled_u8 = (sampled_2d != 0).astype(np.uint8)
    _n_labels, labels = cv2.connectedComponents(sampled_u8, connectivity=8)
    flat_labels = labels.reshape(-1)

    obj = flat_labels[inb]
    pos = obj != 0
    if not np.any(pos):
        return

    x_pos = x[pos].astype(np.int64, copy=False)
    y_pos = y[pos].astype(np.int64, copy=False)
    z_pos = z[pos].astype(np.int64, copy=False)
    voxel_lin = np.ravel_multi_index(
        (x_pos, y_pos, z_pos),
        dims=out_shape,
        mode="raise",
        order="C",
    ).astype(np.int64, copy=False)

    obj_u32 = obj[pos].astype(np.uint64, copy=False)
    sec_u64 = np.uint64(seg_nr)
    obj_key = (sec_u64 << np.uint64(32)) | obj_u32

    pairs = np.empty((voxel_lin.shape[0],), dtype=[("v", "u8"), ("o", "u8")])
    pairs["v"] = voxel_lin.astype(np.uint64, copy=False)
    pairs["o"] = obj_key

    uniq_pairs = np.unique(pairs)
    vox_u = uniq_pairs["v"].astype(np.int64, copy=False)
    vox_ids, per_vox = np.unique(vox_u, return_counts=True)
    np.add.at(ov_flat, vox_ids, per_vox.astype(np.uint32, copy=False))


def _compute_value_volume(gv, fv, ov_flat, value_mode, missing_fill):
    """Derive the value volume from accumulated sums/counts."""
    if value_mode == "mean":
        out = np.zeros_like(gv, dtype=np.float32)
        covered = fv != 0
        out[covered] = gv[covered] / fv[covered].astype(np.float32)
        if missing_fill is not None and np.any(~covered):
            out[~covered] = float(missing_fill)
        return out
    if value_mode == "object_count":
        return ov_flat.reshape(gv.shape).astype(np.float32, copy=False)
    return gv


def _resolve_atlas_mask(use_atlas_mask, atlas_volume, gv):
    """Build the atlas-mask array used during interpolation."""
    if use_atlas_mask and atlas_volume is not None and atlas_volume.shape == gv.shape:
        return atlas_volume != 0
    return None


def _finalize_volumes(
    gv: np.ndarray,
    fv: np.ndarray,
    dv: np.ndarray,
    ov_flat: Optional[np.ndarray],
    vol_cfg: VolumeConfig,
    interp_cfg: InterpolationConfig,
):
    """Convert accumulated sums and optionally interpolate."""
    gv = _compute_value_volume(gv, fv, ov_flat, vol_cfg.value_mode, interp_cfg.missing_fill)

    if interp_cfg.do_interpolation:
        atlas_mask = _resolve_atlas_mask(interp_cfg.use_atlas_mask, interp_cfg.atlas_volume, gv)
        gv = _knn_interpolate_generic(
            gv=gv,
            fv=fv,
            atlas_mask=atlas_mask,
            k=interp_cfg.k,
            batch_size=interp_cfg.batch_size,
            mode="mean",
        )
        if np.any(dv > 0):
            dv_float = dv.astype(np.float32)
            dv_interp = _knn_interpolate_generic(
                gv=dv_float,
                fv=fv,
                atlas_mask=atlas_mask,
                k=interp_cfg.k,
                batch_size=interp_cfg.batch_size,
                mode="max",
            )
            dv = (dv_interp > 0).astype(np.uint8)
    elif interp_cfg.missing_fill is not None and interp_cfg.missing_fill != 0:
        gv[fv == 0] = float(interp_cfg.missing_fill)

    return (
        gv.astype(np.float32, copy=False),
        fv.astype(np.uint32, copy=False),
        dv.astype(np.uint8, copy=False),
    )


def _process_one_section(
    seg_path,
    slice_by_nr,
    vol_cfg: VolumeConfig,
    gv,
    fv,
    dv,
    ov_flat,
):
    """Process a single section path and accumulate into the output volumes."""
    out_shape = gv.shape
    sx, sy, sz = out_shape
    seg_nr = int(number_sections([seg_path])[0])
    slice_info = slice_by_nr.get(seg_nr)
    if not slice_info or not slice_info.anchoring:
        return

    loaded = _read_section_signal(
        seg_path,
        vol_cfg,
    )
    if loaded is None:
        return
    seg_values, mask, seg_height, seg_width = loaded

    reg_height, reg_width = slice_info.height, slice_info.width

    # Prepare damage mask in registration space
    damage_reg = (
        None
        if slice_info.damage_mask is None
        else resize_mask_nearest(
            slice_info.damage_mask.astype(np.uint8),
            reg_width,
            reg_height,
        ).astype(np.uint8)
    )

    # Resample segmentation values into registration space
    src = seg_values if vol_cfg.value_mode == "mean" else mask
    values_reg = cv2.resize(
        src, (reg_width, reg_height), interpolation=cv2.INTER_NEAREST
    )

    # Sample, deform, and remap the plane
    sampled_2d, vals, damage_vals, flat_x, flat_y, plane_h, plane_w = (
        _sample_and_deform_plane(slice_info, values_reg, damage_reg, vol_cfg)
    )

    # Transform flat grid to atlas-space 3-D coordinates
    coords = transform_to_atlas_space(slice_info, flat_y, flat_x)
    if vol_cfg.scale != 1.0:
        coords = coords * float(vol_cfg.scale)

    idx = np.rint(coords).astype(np.int64, copy=False)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    inb = (x >= 0) & (x < sx) & (y >= 0) & (y < sy) & (z >= 0) & (z < sz)
    if not np.any(inb):
        return

    x, y, z = x[inb], y[inb], z[inb]
    np.add.at(fv, (x, y, z), 1)
    if vol_cfg.value_mode != "object_count":
        np.add.at(gv, (x, y, z), vals[inb])

    if damage_vals is not None:
        dv[x, y, z] |= damage_vals[inb].astype(np.uint8)

    if ov_flat is not None:
        _accumulate_object_counts(sampled_2d, inb, x, y, z, seg_nr, out_shape, ov_flat)


def interpolate_volume(
    *,
    segmentation_folder: str,
    alignment_json: str,
    colour,
    atlas: object,
    scale: float = 1.0,
    missing_fill: float = np.nan,
    do_interpolation: bool = True,
    k: int = 5,
    batch_size: int = 200_000,
    use_atlas_mask: bool = True,
    non_linear: bool = True,
    value_mode: str = "pixel_count",
    segmentation_format: str = "binary",
    segmentation_mode: bool = True,
    intensity_channel: str = "grayscale",
    min_intensity: Optional[int] = None,
    max_intensity: Optional[int] = None,
    return_orientation: str = "asr",
):
    """Project section data into atlas-space volumes.

    Parameters
    ----------
    segmentation_folder
        Path to the folder containing segmentation images or source images.
    alignment_json
        Path to the registration JSON passed to
        :func:`PyNutil.read_alignment`.
    colour
        Segmentation color or class identifier to extract. Use ``None`` or
        ``"auto"`` to defer selection to the segmentation adapter.
    atlas
        Atlas definition used to determine the target volume shape. This may
        be a BrainGlobe atlas object or :class:`~PyNutil.AtlasData`.
    scale
        Isotropic scaling factor applied to the atlas output shape.
    missing_fill
        Fill value assigned to voxels with no sampled data when interpolation
        is disabled or when uncovered voxels remain after processing.
    do_interpolation
        If ``True``, fill uncovered voxels using k-nearest-neighbor
        interpolation.
    k
        Number of neighbors to use during interpolation.
    batch_size
        Number of query voxels processed per interpolation batch.
    use_atlas_mask
        If ``True``, restrict interpolation to voxels inside the atlas mask.
    non_linear
        If ``True``, apply non-linear deformation from the registration data
        when available.
    value_mode
        Output volume mode. Supported values are ``"pixel_count"``,
        ``"mean"``, and ``"object_count"``.
    segmentation_format
        Name of the segmentation adapter to use when ``segmentation_mode`` is
        enabled.
    segmentation_mode
        If ``True``, treat input files as segmentation outputs. If ``False``,
        treat them as source images and derive intensities from
        ``intensity_channel``.
    intensity_channel
        Image channel to convert to intensity values when
        ``segmentation_mode=False``.
    min_intensity
        Optional lower threshold for intensity-mode inputs.
    max_intensity
        Optional upper threshold for intensity-mode inputs.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A tuple ``(interpolated_volume, frequency_volume, damage_volume)``.
        The first element stores the requested value volume, the second stores
        per-voxel sampling frequency, and the third is a binary damage mask.

    Examples
    --------
    Build atlas-space volumes from segmentation images:

    >>> gv, fv, dv = interpolate_volume(
    ...     segmentation_folder="path/to/segmentations/",
    ...     alignment_json="path/to/alignment.json",
    ...     colour=[0, 0, 0],
    ...     atlas=atlas,
    ... )
    """
    if value_mode not in {"pixel_count", "mean", "object_count"}:
        raise ValueError(
            "value_mode must be one of 'pixel_count', 'mean', or 'object_count'"
        )

    if hasattr(atlas, "annotation") and getattr(atlas, "annotation") is not None:
        atlas_volume = getattr(atlas, "annotation")
    elif hasattr(atlas, "volume") and getattr(atlas, "volume") is not None:
        atlas_volume = getattr(atlas, "volume")
    else:
        raise ValueError(
            "atlas must provide a non-None 'annotation' or 'volume' attribute"
        )

    out_base_shape = tuple(int(x) for x in atlas_volume.shape)

    out_shape = derive_shape_from_atlas(atlas_shape=out_base_shape, scale=scale)

    registration = read_alignment(alignment_json)
    slice_by_nr = {s.section_number: s for s in registration.slices}
    seg_paths = discover_image_files(segmentation_folder)

    # Accept GUI/settings values like "auto" and defer to adapter auto-detection
    # by passing pixel_id=None.
    if isinstance(colour, str):
        colour_str = colour.strip()
        if colour_str.lower() == "auto" or colour_str == "":
            colour_arr = None
        elif colour_str.isdigit():
            colour_arr = np.array([int(colour_str)], dtype=np.uint8)
        elif "," in colour_str:
            colour_arr = np.array(
                [int(x.strip()) for x in colour_str.strip("[]").split(",") if x.strip()],
                dtype=np.uint8,
            )
        else:
            raise ValueError(
                "colour must be None, 'auto', an int-like string, or a list/tuple of ints"
            )
    else:
        colour_arr = np.array(colour, dtype=np.uint8) if colour is not None else None
    vol_cfg = VolumeConfig(
        segmentation_adapter=(
            SegmentationAdapterRegistry.get(segmentation_format)
            if segmentation_mode
            else None
        ),
        segmentation_mode=segmentation_mode,
        colour_arr=colour_arr,
        intensity_channel=intensity_channel,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        scale=scale,
        non_linear=non_linear,
        value_mode=value_mode,
    )
    interp_cfg = InterpolationConfig(
        do_interpolation=do_interpolation,
        missing_fill=missing_fill,
        use_atlas_mask=use_atlas_mask,
        atlas_volume=atlas_volume,
        k=k,
        batch_size=batch_size,
    )

    gv = np.zeros(out_shape, dtype=np.float32)
    fv = np.zeros(out_shape, dtype=np.uint32)
    dv = np.zeros(out_shape, dtype=np.uint8)
    ov_flat = (
        np.zeros((gv.size,), dtype=np.uint32) if value_mode == "object_count" else None
    )

    for seg_path in seg_paths:
        _process_one_section(seg_path, slice_by_nr, vol_cfg, gv, fv, dv, ov_flat)

    gv, fv, dv = _finalize_volumes(gv, fv, dv, ov_flat, vol_cfg, interp_cfg)

    if return_orientation != "lpi":
        from .reorientation import reorient_volume
        atlas_shape = out_base_shape
        gv = reorient_volume(gv, atlas_shape, return_orientation)
        fv = reorient_volume(fv, atlas_shape, return_orientation)
        dv = reorient_volume(dv, atlas_shape, return_orientation)

    return gv, fv, dv

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from .adapters import load_registration
from .transforms import transform_to_atlas_space
from .utils import number_sections, convert_to_intensity, discover_image_files


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
    colour_arr: Optional[np.ndarray],
    intensity_channel: str,
    min_intensity: Optional[int],
    max_intensity: Optional[int],
):
    """Load an image and return (seg_values, mask, seg_height, seg_width) or None."""
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None

    if colour_arr is None:
        # Intensity mode
        seg_values = convert_to_intensity(seg, intensity_channel)
        if min_intensity is not None:
            seg_values[seg_values < min_intensity] = 0
        if max_intensity is not None:
            seg_values[seg_values > max_intensity] = 0
        mask = (seg_values != 0).astype(np.float32, copy=False)
    else:
        # Segmentation mode
        if seg.ndim == 2:
            seg_values = seg.astype(np.float32, copy=False)
            mask = (seg_values != 0).astype(np.float32, copy=False)
        else:
            seg = seg[:, :, :3]
            mask = np.all(seg == colour_arr[None, None, :], axis=2).astype(np.float32)
            seg_values = mask

    seg_height, seg_width = seg.shape[:2]
    return seg_values, mask, seg_height, seg_width


def _sample_and_deform_plane(
    slice_info,
    values_reg: np.ndarray,
    damage_reg: Optional[np.ndarray],
    scale: float,
    reg_height: int,
    reg_width: int,
    non_linear: bool,
):
    """Construct a sampling grid, optionally deform, and remap values.

    Returns (sampled_2d, vals_flat, damage_vals, flat_x, flat_y, plane_h, plane_w).
    """
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

    if non_linear and slice_info.forward_deformation is not None:
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


def _compute_value_volume(gv, fv, ov_flat, out_shape, value_mode, missing_fill):
    """Derive the value volume from accumulated sums/counts."""
    if value_mode == "mean":
        out = np.zeros_like(gv, dtype=np.float32)
        covered = fv != 0
        out[covered] = gv[covered] / fv[covered].astype(np.float32)
        if missing_fill is not None and np.any(~covered):
            out[~covered] = float(missing_fill)
        return out
    if value_mode == "object_count":
        return ov_flat.reshape(out_shape).astype(np.float32, copy=False)
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
    out_shape: Tuple[int, int, int],
    value_mode: str,
    missing_fill: float,
    do_interpolation: bool,
    atlas_volume: Optional[np.ndarray],
    use_atlas_mask: bool,
    k: int,
    batch_size: int,
):
    """Convert accumulated sums and optionally interpolate."""
    gv = _compute_value_volume(gv, fv, ov_flat, out_shape, value_mode, missing_fill)

    if do_interpolation:
        atlas_mask = _resolve_atlas_mask(use_atlas_mask, atlas_volume, gv)
        gv = _knn_interpolate_generic(
            gv=gv,
            fv=fv,
            atlas_mask=atlas_mask,
            k=k,
            batch_size=batch_size,
            mode="mean",
        )
        if np.any(dv > 0):
            dv_float = dv.astype(np.float32)
            dv_interp = _knn_interpolate_generic(
                gv=dv_float,
                fv=fv,
                atlas_mask=atlas_mask,
                k=k,
                batch_size=batch_size,
                mode="max",
            )
            dv = (dv_interp > 0).astype(np.uint8)
    elif missing_fill is not None and missing_fill != 0:
        gv[fv == 0] = float(missing_fill)

    return (
        gv.astype(np.float32, copy=False),
        fv.astype(np.uint32, copy=False),
        dv.astype(np.uint8, copy=False),
    )


def _process_one_section(
    seg_path,
    slice_by_nr,
    colour_arr,
    intensity_channel,
    min_intensity,
    max_intensity,
    scale,
    non_linear,
    value_mode,
    gv,
    fv,
    dv,
    ov_flat,
    out_shape,
):
    """Process a single section path and accumulate into the output volumes."""
    sx, sy, sz = out_shape
    seg_nr = int(number_sections([seg_path])[0])
    slice_info = slice_by_nr.get(seg_nr)
    if not slice_info or not slice_info.anchoring:
        return

    loaded = _read_section_signal(
        seg_path, colour_arr, intensity_channel, min_intensity, max_intensity
    )
    if loaded is None:
        return
    seg_values, mask, seg_height, seg_width = loaded

    reg_height, reg_width = slice_info.height, slice_info.width

    # Prepare damage mask in registration space
    damage_reg = _resize_damage_mask(slice_info.damage_mask, reg_width, reg_height)

    # Resample segmentation values into registration space
    src = seg_values if value_mode == "mean" else mask
    values_reg = cv2.resize(
        src, (reg_width, reg_height), interpolation=cv2.INTER_NEAREST
    )

    # Sample, deform, and remap the plane
    sampled_2d, vals, damage_vals, flat_x, flat_y, plane_h, plane_w = (
        _sample_and_deform_plane(
            slice_info,
            values_reg,
            damage_reg,
            scale,
            reg_height,
            reg_width,
            non_linear,
        )
    )

    # For brainglobe registration, apply the deformation to correct the
    # atlas-slice positions before 3D transformation.  The brain section has
    # different dimensions than the atlas slice; without this correction
    # pixels are placed at linearly-scaled positions that can be tens of
    # voxels off (typically pushing edges outside the brain).
    if (
        non_linear
        and slice_info.deformation is not None
        and slice_info.metadata.get("registration_type") == "brainglobe"
    ):
        corrected_x, corrected_y = slice_info.deformation(
            flat_x.astype(np.float64), flat_y.astype(np.float64)
        )
        coords = transform_to_atlas_space(
            slice_info.anchoring, corrected_y, corrected_x, reg_height, reg_width
        )
    else:
        coords = transform_to_atlas_space(
            slice_info.anchoring, flat_y, flat_x, reg_height, reg_width
        )
    if scale != 1.0:
        coords = coords * float(scale)

    idx = np.rint(coords).astype(np.int64, copy=False)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    inb = (x >= 0) & (x < sx) & (y >= 0) & (y < sy) & (z >= 0) & (z < sz)
    if not np.any(inb):
        return

    x, y, z = x[inb], y[inb], z[inb]
    np.add.at(fv, (x, y, z), 1)
    np.add.at(gv, (x, y, z), vals[inb])

    if damage_vals is not None:
        dv[x, y, z] |= damage_vals[inb].astype(np.uint8)

    if ov_flat is not None:
        _accumulate_object_counts(sampled_2d, inb, x, y, z, seg_nr, out_shape, ov_flat)


def _resize_damage_mask(damage_mask, reg_width, reg_height):
    """Resize a damage mask to registration resolution, or return None."""
    if damage_mask is None:
        return None
    if damage_mask.shape != (reg_height, reg_width):
        return cv2.resize(
            damage_mask.astype(np.uint8),
            (reg_width, reg_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return damage_mask.astype(np.uint8)


def project_sections_to_volume(
    *,
    segmentation_folder: str,
    alignment_json: str,
    colour,
    atlas_shape: Tuple[int, int, int],
    atlas_volume: Optional[np.ndarray],
    scale: float = 1.0,
    missing_fill: float = np.nan,
    do_interpolation: bool = True,
    k: int = 5,
    batch_size: int = 200_000,
    use_atlas_mask: bool = True,
    non_linear: bool = True,
    value_mode: str = "pixel_count",
    intensity_channel: str = "grayscale",
    min_intensity: Optional[int] = None,
    max_intensity: Optional[int] = None,
):
    """Project section segmentations into a 3D atlas-space volume.

    Constructs three volumes:
        - value volume (gv): depends on *value_mode*
        - frequency volume (fv): number of sampled pixels per voxel
        - damage volume (dv): binary mask of damaged voxels

    Supported *value_mode* values: ``"pixel_count"``, ``"mean"``, ``"object_count"``.
    """
    if value_mode not in {"pixel_count", "mean", "object_count"}:
        raise ValueError(
            "value_mode must be one of 'pixel_count', 'mean', or 'object_count'"
        )

    out_shape = derive_shape_from_atlas(atlas_shape=atlas_shape, scale=scale)

    registration = load_registration(alignment_json)
    slice_by_nr = {s.section_number: s for s in registration.slices}
    seg_paths = discover_image_files(segmentation_folder)

    colour_arr = np.array(colour, dtype=np.uint8) if colour is not None else None

    gv = np.zeros(out_shape, dtype=np.float32)
    fv = np.zeros(out_shape, dtype=np.uint32)
    dv = np.zeros(out_shape, dtype=np.uint8)
    ov_flat = (
        np.zeros((gv.size,), dtype=np.uint32) if value_mode == "object_count" else None
    )

    for seg_path in seg_paths:
        _process_one_section(
            seg_path,
            slice_by_nr,
            colour_arr,
            intensity_channel,
            min_intensity,
            max_intensity,
            scale,
            non_linear,
            value_mode,
            gv,
            fv,
            dv,
            ov_flat,
            out_shape,
        )

    return _finalize_volumes(
        gv,
        fv,
        dv,
        ov_flat,
        out_shape,
        value_mode,
        missing_fill,
        do_interpolation,
        atlas_volume,
        use_atlas_mask,
        k,
        batch_size,
    )

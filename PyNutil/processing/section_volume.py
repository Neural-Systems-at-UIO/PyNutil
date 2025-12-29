from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..io.read_and_write import load_quint_json
from .transformations import transform_to_atlas_space
from .utils import number_sections, convert_to_intensity, create_damage_mask
from .visualign_deformations import triangulate, transform_vec, forwardtransform_vec


def derive_shape_from_atlas(
    *,
    atlas_shape: Tuple[int, int, int],
    scale: float,
) -> Tuple[int, int, int]:
    """Derive an output shape from atlas shape + scale."""

    if scale <= 0:
        raise ValueError("scale must be > 0")

    return tuple(max(1, int(round(int(s) * float(scale)))) for s in atlas_shape)


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
    out_vals = np.empty((query_pts.shape[0],), dtype=np.float32)

    for start in range(0, query_pts.shape[0], batch_size):
        end = min(start + batch_size, query_pts.shape[0])
        q = query_pts[start:end]
        dist, ind = tree.query(q, k=k)
        if k == 1:
            out_vals[start:end] = fit_vals[ind]
        else:
            neigh_vals = fit_vals[ind]
            if mode == "mean":
                out_vals[start:end] = neigh_vals.mean(axis=1)
            elif mode == "max":
                out_vals[start:end] = neigh_vals.max(axis=1)

    if atlas_mask is not None:
        out = np.zeros_like(gv)
        out[target_mask] = out_vals
        return out

    gv[target_mask] = out_vals
    return gv


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

    This constructs two 3D volumes:
        - value volume (gv): depends on `value_mode`
        - frequency volume (fv): denominator used by `value_mode` (see below)

    Supported `value_mode`:
        - "pixel_count": number of segmented pixels per voxel (default; backwards compatible)
        - "mean": mean segmentation value per voxel, averaged over all sampled pixels (including zeros)
        - "object_count": number of 2D connected components contributing to each voxel

    Notes:
        - `fv` always counts all sampled pixels per voxel.

    Interpolation (if enabled) is applied to the value volume; the frequency
    volume is never interpolated.
    """

    import cv2
    import os

    out_shape = derive_shape_from_atlas(atlas_shape=atlas_shape, scale=scale)

    quint_json = load_quint_json(alignment_json)
    slices = quint_json["slices"]
    slice_by_nr = {int(s.get("nr")): s for s in slices if s.get("nr") is not None}
    grid_spacing = quint_json.get("gridspacing")

    seg_paths = []
    for name in os.listdir(segmentation_folder):
        p = os.path.join(segmentation_folder, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"} or name.endswith(
            ".dzip"
        ):
            seg_paths.append(p)
    seg_paths.sort()

    gv = np.zeros(out_shape, dtype=np.float32)
    fv = np.zeros(out_shape, dtype=np.uint32)
    dv = np.zeros(out_shape, dtype=np.uint8)

    if value_mode not in {"pixel_count", "mean", "object_count"}:
        raise ValueError(
            "value_mode must be one of 'pixel_count', 'mean', or 'object_count'"
        )
    ov_flat: Optional[np.ndarray]
    if value_mode == "object_count":
        ov_flat = np.zeros((gv.size,), dtype=np.uint32)
    else:
        ov_flat = None

    sx, sy, sz = out_shape
    if colour is not None:
        colour_arr = np.array(colour, dtype=np.uint8)
    else:
        colour_arr = None

    for seg_path in seg_paths:
        seg_nr = int(number_sections([seg_path])[0])
        slice_dict = slice_by_nr.get(seg_nr)
        if not slice_dict or not slice_dict.get("anchoring"):
            continue

        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg is None:
            continue

        if colour_arr is None:
            # Intensity mode
            seg_values = convert_to_intensity(seg, intensity_channel)
            if min_intensity is not None:
                seg_values[seg_values < min_intensity] = 0
            if max_intensity is not None:
                seg_values[seg_values > max_intensity] = 0
            mask = (seg_values != 0).astype(np.float32, copy=False)
            seg_height, seg_width = seg.shape[:2]
        else:
            # Segmentation mode
            if seg.ndim == 2:
                seg_values = seg.astype(np.float32, copy=False)
                mask = (seg_values != 0).astype(np.float32, copy=False)
                seg_height, seg_width = seg.shape
            else:
                seg = seg[:, :, :3]
                mask = np.all(seg == colour_arr[None, None, :], axis=2).astype(np.float32)
                # For RGB segmentations, we only currently support colour-matched binary masks.
                seg_values = mask
                seg_height, seg_width = seg.shape[:2]

        reg_height, reg_width = int(slice_dict["height"]), int(slice_dict["width"])

        # Handle damage mask if present
        damage_reg = None
        if grid_spacing is not None and "grid" in slice_dict:
            damage_mask = create_damage_mask(slice_dict, grid_spacing)
            damage_reg = cv2.resize(
                (damage_mask == 0).astype(np.uint8),
                (reg_width, reg_height),
                interpolation=cv2.INTER_NEAREST,
            )

        # Plane sampling: evaluate on a regular grid sized to the slice plane
        # extent in atlas voxels (||u|| and ||v||), similar to the reference
        # "perfect_image" approach but without atlas-specific assumptions.
        anch = slice_dict["anchoring"]
        u = np.asarray(anch[3:6], dtype=np.float32)
        v = np.asarray(anch[6:9], dtype=np.float32)
        plane_w = max(1, int(round(float(np.linalg.norm(u)) * float(scale))))
        plane_h = max(1, int(round(float(np.linalg.norm(v)) * float(scale))))

        # Resample segmentation into registration space.
        # For mean mode, we use the raw segmentation values (2D scalar images),
        # but compute the mean over non-zero values only.
        if value_mode == "mean":
            values_reg = cv2.resize(
                seg_values, (reg_width, reg_height), interpolation=cv2.INTER_NEAREST
            )
        else:
            values_reg = cv2.resize(
                mask, (reg_width, reg_height), interpolation=cv2.INTER_NEAREST
            )

        yy, xx = np.indices((plane_h, plane_w), dtype=np.float32)
        reg_x = (xx + 0.5) * (float(reg_width) / float(plane_w))
        reg_y = (yy + 0.5) * (float(reg_height) / float(plane_h))

        flat_x = reg_x.reshape(-1)
        flat_y = reg_y.reshape(-1)

        if non_linear and "markers" in slice_dict:
            tri = triangulate(reg_width, reg_height, slice_dict["markers"])
            new_x, new_y = forwardtransform_vec(tri, flat_x, flat_y)
            map_x = new_x.reshape((plane_h, plane_w)).astype(np.float32, copy=False)
            map_y = new_y.reshape((plane_h, plane_w)).astype(np.float32, copy=False)
        else:
            map_x = reg_x.astype(np.float32, copy=False)
            map_y = reg_y.astype(np.float32, copy=False)
            new_x = flat_x
            new_y = flat_y

        sampled = cv2.remap(
            values_reg,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        vals = sampled.reshape(-1).astype(np.float32, copy=False)

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

        flat_labels: Optional[np.ndarray] = None
        if ov_flat is not None:
            sampled_u8 = (sampled != 0).astype(np.uint8)
            _n_labels, labels = cv2.connectedComponents(sampled_u8, connectivity=8)
            flat_labels = labels.reshape(-1)

        coords = transform_to_atlas_space(
            slice_dict["anchoring"], flat_y, flat_x, reg_height, reg_width
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
        vals_in = vals[inb]

        np.add.at(fv, (x, y, z), 1)
        np.add.at(gv, (x, y, z), vals_in)

        if damage_vals is not None:
            damage_in = damage_vals[inb]
            # Mark voxels as damaged if any contributing pixel is damaged
            dv[x, y, z] |= damage_in.astype(np.uint8)

        if ov_flat is not None:
            if flat_labels is None:  # pragma: no cover
                continue
            obj = flat_labels[inb]
            pos = obj != 0
            if np.any(pos):
                x_pos = x[pos].astype(np.int64, copy=False)
                y_pos = y[pos].astype(np.int64, copy=False)
                z_pos = z[pos].astype(np.int64, copy=False)
                voxel_lin = np.ravel_multi_index(
                    (x_pos, y_pos, z_pos), dims=out_shape, mode="raise", order="C"
                ).astype(np.int64, copy=False)

                obj_u32 = obj[pos].astype(np.uint64, copy=False)
                sec_u64 = np.uint64(seg_nr)
                obj_key = (sec_u64 << np.uint64(32)) | obj_u32

                pairs = np.empty(
                    (voxel_lin.shape[0],), dtype=[("v", "u8"), ("o", "u8")]
                )
                pairs["v"] = voxel_lin.astype(np.uint64, copy=False)
                pairs["o"] = obj_key

                uniq_pairs = np.unique(pairs)
                vox_u = uniq_pairs["v"].astype(np.int64, copy=False)
                vox_ids, per_vox = np.unique(vox_u, return_counts=True)
                np.add.at(ov_flat, vox_ids, per_vox.astype(np.uint32, copy=False))

    # Convert accumulated sums into the requested value volume.
    if value_mode == "mean":
        out = np.zeros_like(gv, dtype=np.float32)
        covered = fv != 0
        out[covered] = gv[covered] / fv[covered].astype(np.float32)
        if missing_fill is not None and np.any(~covered):
            out[~covered] = float(missing_fill)
        gv = out
    elif value_mode == "object_count":
        gv = ov_flat.reshape(out_shape).astype(np.float32, copy=False)

    if do_interpolation:
        atlas_mask = None
        if use_atlas_mask and atlas_volume is not None and atlas_volume.shape == gv.shape:
            atlas_mask = atlas_volume != 0

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
    else:
        if missing_fill is not None and not (missing_fill == 0):
            gv[fv == 0] = float(missing_fill)

    return (
        gv.astype(np.float32, copy=False),
        fv.astype(np.uint32, copy=False),
        dv.astype(np.uint8, copy=False),
    )



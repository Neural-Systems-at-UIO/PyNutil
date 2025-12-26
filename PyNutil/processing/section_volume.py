from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..io.read_and_write import load_quint_json
from .transformations import transform_to_atlas_space, transform_to_registration
from .utils import number_sections
from .visualign_deformations import triangulate, transform_vec


def derive_shape_from_atlas(
    *,
    atlas_shape: Tuple[int, int, int],
    scale: float,
    shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[int, int, int]:
    """Derive an output shape from atlas shape + scale.

    `shape` is a deprecated escape hatch. If provided, `scale` must be 1.
    """

    if shape is not None:
        if scale != 1.0:
            raise ValueError(
                "Do not pass both shape and scale; shape is derived from scale and atlas shape."
            )
        return tuple(int(x) for x in shape)

    if scale <= 0:
        raise ValueError("scale must be > 0")

    return tuple(max(1, int(round(int(s) * float(scale)))) for s in atlas_shape)


def _knn_interpolate_generic(
    *,
    gv: np.ndarray,
    fv: np.ndarray,
    atlas_mask: Optional[np.ndarray],
    k: int,
    weights: str,
    batch_size: int,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SciPy is required for do_interpolation=True") from exc

    if k < 1:
        raise ValueError("k must be >= 1")
    if k > 1 and weights not in {"uniform", "distance"}:
        raise ValueError("weights must be 'uniform' or 'distance'")

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
    shape: Optional[Tuple[int, int, int]] = None,
    missing_fill: float = np.nan,
    do_interpolation: bool = True,
    k: int = 5,
    weights: str = "uniform",
    batch_size: int = 200_000,
    use_atlas_mask: bool = True,
    non_linear: bool = True,
):
    """Project section segmentations into a 3D atlas-space volume.

    This constructs two 3D volumes:
        - signal volume (gv): sum of per-pixel mask values (0/1)
        - frequency volume (fv): number of contributing pixels per voxel

    Interpolation (if enabled) is applied to the signal volume; the frequency
    volume is never interpolated.
    """

    import cv2
    import os

    out_shape = derive_shape_from_atlas(atlas_shape=atlas_shape, scale=scale, shape=shape)

    quint_json = load_quint_json(alignment_json)
    slices = quint_json["slices"]
    slice_by_nr = {int(s.get("nr")): s for s in slices if s.get("nr") is not None}

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

    sx, sy, sz = out_shape
    colour_arr = np.array(colour, dtype=np.uint8)

    for seg_path in seg_paths:
        seg_nr = int(number_sections([seg_path])[0])
        slice_dict = slice_by_nr.get(seg_nr)
        if not slice_dict or not slice_dict.get("anchoring"):
            continue

        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg is None:
            continue
        if seg.ndim == 2:
            mask = (seg != 0).astype(np.float32)
            seg_height, seg_width = seg.shape
        else:
            seg = seg[:, :, :3]
            mask = np.all(seg == colour_arr[None, None, :], axis=2).astype(np.float32)
            seg_height, seg_width = seg.shape[:2]

        reg_height, reg_width = int(slice_dict["height"]), int(slice_dict["width"])
        y_scale, x_scale = transform_to_registration(
            seg_height, seg_width, reg_height, reg_width
        )

        yy, xx = np.indices((seg_height, seg_width), dtype=np.float32)
        scaled_y = yy * float(y_scale)
        scaled_x = xx * float(x_scale)

        if non_linear and "markers" in slice_dict:
            tri = triangulate(reg_width, reg_height, slice_dict["markers"])
            flat_x = scaled_x.reshape(-1)
            flat_y = scaled_y.reshape(-1)
            new_x, new_y = transform_vec(tri, flat_x, flat_y)
        else:
            new_x = scaled_x.reshape(-1)
            new_y = scaled_y.reshape(-1)

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

        np.add.at(fv, (x, y, z), 1)
        np.add.at(gv, (x, y, z), vals)

    if do_interpolation:
        atlas_mask = None
        if use_atlas_mask and atlas_volume is not None and atlas_volume.shape == gv.shape:
            atlas_mask = atlas_volume != 0

        gv = _knn_interpolate_generic(
            gv=gv,
            fv=fv,
            atlas_mask=atlas_mask,
            k=k,
            weights=weights,
            batch_size=batch_size,
        )

        if atlas_mask is not None:
            out = np.zeros_like(gv)
            out[atlas_mask] = gv[atlas_mask]
            gv = out
    else:
        if missing_fill is not None and not (missing_fill == 0):
            gv[fv == 0] = float(missing_fill)

    return gv.astype(np.float32, copy=False), fv.astype(np.uint32, copy=False)


# Backwards-compatible name used by the public PyNutil API wrapper.
interpolate_volume = project_sections_to_volume

"""Atlas map construction and region-area computation.

This module groups every function that turns an anchoring vector (or
flat-file path) into a 2-D atlas map and derives per-region statistics
from it.

Public API
----------
- :func:`generate_target_slice` — extract a 2-D slice from a 3-D atlas.
- :func:`warp_image` — apply non-linear deformation to an image.
- :func:`load_atlas_image` — load / generate an atlas map for a section.
- :func:`flat_to_dataframe` — count region pixels with optional damage/hemi masks.
- :func:`transform_to_atlas_space` — transform 2-D coordinates to 3-D atlas space.
- :func:`get_region_areas` — build atlas map and compute region areas for a section.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd

from .utils import resize_mask_nearest


# ---------------------------------------------------------------------------
# Slice extraction from 3-D atlas
# ---------------------------------------------------------------------------


def _compute_slice_coordinates(ouv, ref_shape):
    """Compute 2D-to-3D index arrays for a given anchoring vector.

    Args:
        ouv (list): Orientation vector [ox, oy, oz, ux, uy, uz, vx, vy, vz].
        ref_shape (tuple): Shape of the reference 3D volume (xdim, ydim, zdim).

    Returns:
        (lx, ly, lz, valid, height, width) — index arrays and validity mask.
    """
    ox, oy, oz, ux, uy, uz, vx, vy, vz = ouv
    width = int(np.floor(math.hypot(ux, uy, uz))) + 1
    height = int(np.floor(math.hypot(vx, vy, vz))) + 1
    xdim, ydim, zdim = ref_shape

    yf = np.arange(height, dtype=np.float64) / height
    xf = np.arange(width, dtype=np.float64) / width

    lx = np.floor(ox + (vx * yf)[:, None] + ux * xf).astype(np.int32)
    ly = np.floor(oy + (vy * yf)[:, None] + uy * xf).astype(np.int32)
    lz = np.floor(oz + (vz * yf)[:, None] + uz * xf).astype(np.int32)

    valid = (
        (lx >= 0) & (lx < xdim)
        & (ly >= 0) & (ly < ydim)
        & (lz >= 0) & (lz < zdim)
    )
    return lx, ly, lz, valid, height, width


def generate_target_slice(ouv, atlas):
    """Generate a 2D slice from a 3D atlas based on orientation vectors.

    Args:
        ouv (list): Orientation vector [ox, oy, oz, ux, uy, uz, vx, vy, vz].
        atlas (ndarray): 3D atlas volume.

    Returns:
        ndarray: 2D slice extracted from the atlas.
    """
    lx, ly, lz, valid, height, width = _compute_slice_coordinates(ouv, atlas.shape)
    data_im = np.zeros((height, width), dtype=np.uint32)
    data_im[valid] = atlas[lx[valid], ly[valid], lz[valid]]
    return data_im


def generate_target_slices(ouv, *volumes):
    """Extract 2D slices from multiple 3D volumes sharing the same coordinates.

    All volumes must have the same shape. The coordinate arrays are computed
    once and reused for every volume, saving ~50% of the work compared to
    calling :func:`generate_target_slice` separately for each volume.

    Args:
        ouv (list): Orientation vector [ox, oy, oz, ux, uy, uz, vx, vy, vz].
        *volumes: One or more 3D ndarrays with identical shape.

    Returns:
        tuple of ndarrays: One 2D slice per input volume.
    """
    ref_shape = volumes[0].shape
    lx, ly, lz, valid, height, width = _compute_slice_coordinates(ouv, ref_shape)
    lx_v, ly_v, lz_v = lx[valid], ly[valid], lz[valid]

    results = []
    for vol in volumes:
        data_im = np.zeros((height, width), dtype=np.uint32)
        data_im[valid] = vol[lx_v, ly_v, lz_v]
        results.append(data_im)
    return tuple(results)


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------


def compute_deformation_map(image_shape, deformation, rescaleXY):
    """Precompute the deformation coordinate map for a given image shape.

    The returned map can be reused to warp multiple images of the same
    shape with :func:`apply_deformation_map`, avoiding redundant calls
    to the (expensive) deformation function.

    Args:
        image_shape (tuple): (height, width) of the source image.
        deformation (callable): Deformation function (x, y) → (new_x, new_y).
        rescaleXY (tuple or None): (width, height) for rescaling.

    Returns:
        tuple: (newY, newX, oob) — int32 index arrays and out-of-bounds mask.
    """
    reg_h, reg_w = image_shape
    if rescaleXY is not None:
        w, h = rescaleXY
    else:
        w, h = reg_w, reg_h
    oldX, oldY = np.meshgrid(np.arange(reg_w), np.arange(reg_h))
    w_scale = w / reg_w
    h_scale = h / reg_h
    newX, newY = deformation(oldX.ravel() * w_scale, oldY.ravel() * h_scale)
    newX = (newX / w_scale).reshape(reg_h, reg_w).astype(np.int32)
    newY = (newY / h_scale).reshape(reg_h, reg_w).astype(np.int32)
    oob = (newX < 0) | (newX >= reg_w) | (newY < 0) | (newY >= reg_h)
    # Pre-clip so apply_deformation_map can index directly.
    np.clip(newX, 0, reg_w - 1, out=newX)
    np.clip(newY, 0, reg_h - 1, out=newY)
    return newY, newX, oob


def apply_deformation_map(image, deform_map):
    """Warp *image* using a precomputed deformation map.

    Args:
        image (ndarray): 2D image to warp (same shape used to create the map).
        deform_map (tuple): (newY, newX, oob) from :func:`compute_deformation_map`.

    Returns:
        ndarray: Warped image.
    """
    newY, newX, oob = deform_map
    new_image = image[newY, newX]
    new_image[oob] = 0
    return new_image


def warp_image(image, deformation, rescaleXY):
    """Warp an image using a deformation function, applying optional resizing.

    Args:
        image (ndarray): Image array to be warped.
        deformation (callable): Deformation function that takes (x, y) arrays
            and returns (new_x, new_y) arrays.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        ndarray: The warped image array.
    """
    deform_map = compute_deformation_map(image.shape, deformation, rescaleXY)
    return apply_deformation_map(image, deform_map)


# ---------------------------------------------------------------------------
# Label assignment and loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _read_itksnap_label_lookup(path):
    """Read a label lookup file and return ordered atlas IDs by label index."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if "idx" in df.columns:
            return df["idx"].to_numpy(dtype=np.int64)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(
            dtype=np.int64
        )

    ids = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t") if "\t" in s else s.split()
            try:
                ids.append(int(parts[0]))
            except (ValueError, IndexError):
                continue
    return np.asarray(ids, dtype=np.int64)


def load_atlas_image(
    image_vector,
    volume,
    deformation,
    rescaleXY,
    deform_map=None,
    precomputed_slice=None,
):
    """Generate a 2D atlas slice from a 3D atlas volume, optionally warping it."""
    if precomputed_slice is not None:
        image = np.float64(precomputed_slice)
    else:
        image = np.float64(generate_target_slice(image_vector, volume))
    if deformation is not None or deform_map is not None:
        if deform_map is not None:
            image = apply_deformation_map(image, deform_map)
        else:
            image = warp_image(image, deformation, rescaleXY)
    return image


# ---------------------------------------------------------------------------
# Region counting from atlas map
# ---------------------------------------------------------------------------

def flat_to_dataframe(image, damage_mask, hemi_mask, rescaleXY=None):
    """Build a DataFrame from an atlas map, with optional damage/hemisphere masks.

    Uses a single ``np.unique`` / ``np.bincount`` pass to count pixels for
    all (label × hemi × damage) combinations at once, instead of calling
    ``np.unique`` separately for each combo.

    Args:
        image (ndarray): Source image with label IDs.
        damage_mask (ndarray): Binary mask indicating damaged areas.
        hemi_mask (ndarray): Binary mask for hemisphere assignment.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        DataFrame: Pixel counts grouped by label.
    """
    scale_factor = (
        (rescaleXY[0] * rescaleXY[1]) / (image.shape[0] * image.shape[1])
        if rescaleXY
        else None
    )
    if hemi_mask is not None:
        hemi_mask = resize_mask_nearest(
            hemi_mask.astype(np.uint8), image.shape[1], image.shape[0]
        )

    if damage_mask is not None:
        damage_mask = resize_mask_nearest(
            damage_mask.astype(np.uint8), image.shape[1], image.shape[0]
        ).astype(bool)

    # --- single-pass counting via np.unique + np.bincount ---
    flat_img = image.ravel()
    unique_ids, inverse = np.unique(flat_img, return_inverse=True)

    # Build the list of (hemi_val, damage_val, column_name) combos
    if (hemi_mask is not None) and (damage_mask is not None):
        combos = [
            (1, 0, "left_hemi_undamaged_region_area"),
            (1, 1, "left_hemi_damaged_region_area"),
            (2, 0, "right_hemi_undamaged_region_area"),
            (2, 1, "right_hemi_damaged_region_area"),
        ]
    elif hemi_mask is not None:
        combos = [(1, 0, "left_hemi_region_area"), (2, 0, "right_hemi_region_area")]
    elif damage_mask is not None:
        combos = [(0, 0, "undamaged_region_area"), (0, 1, "damaged_region_area")]
    else:
        combos = [(None, None, "region_area")]

    data = {"idx": unique_ids}

    for hemi_val, damage_val, col_name in combos:
        mask = np.ones(flat_img.size, dtype=bool)
        if hemi_mask is not None:
            mask &= hemi_mask.ravel() == hemi_val
        if damage_mask is not None:
            mask &= damage_mask.ravel() == damage_val
        # bincount with the inverse mapping; only count where mask is True
        counts = np.bincount(inverse[mask], minlength=len(unique_ids))
        if scale_factor:
            data[col_name] = counts.astype(np.float64) * scale_factor
        else:
            data[col_name] = counts.astype(np.float64)

    df_area_per_label = pd.DataFrame(data)

    # Derive aggregate area columns from leaf-level columns
    if (hemi_mask is not None) and (damage_mask is not None):
        df_area_per_label["undamaged_region_area"] = (
            df_area_per_label["left_hemi_undamaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["damaged_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_damaged_region_area"]
        )
        df_area_per_label["left_hemi_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["left_hemi_undamaged_region_area"]
        )
        df_area_per_label["right_hemi_region_area"] = (
            df_area_per_label["right_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )
    elif hemi_mask is not None:
        df_area_per_label["region_area"] = (
            df_area_per_label["left_hemi_region_area"]
            + df_area_per_label["right_hemi_region_area"]
        )
    elif damage_mask is not None:
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )

    return df_area_per_label


# ---------------------------------------------------------------------------
# Atlas-space coordinate transformation
# ---------------------------------------------------------------------------


def transform_to_atlas_space(
    slice_info,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Transform 2-D registration coordinates to 3-D atlas space.

    Uses the QuickNII anchoring vector to apply the affine:
        atlas_coord = O + (x/width) * U + (y/height) * V

    Args:
        slice_info: SliceInfo with `anchoring` (9-element vector), `height`,
            and `width` for the registration image.
        y: Y coordinates in registration space.
        x: X coordinates in registration space.

    Returns:
        (N, 3) array of 3-D atlas-space coordinates.
    """
    # NOTE: This implementation intentionally avoids building intermediate arrays via
    # np.array([row0, row1, row2]).T, which has been observed to miscompute under
    # some Python/numpy builds for large inputs.
    anchoring = slice_info.anchoring
    o = np.asarray(anchoring[0:3], dtype=np.float64)
    u = np.asarray(anchoring[3:6], dtype=np.float64)
    v = np.asarray(anchoring[6:9], dtype=np.float64)

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_scale = y_arr / float(slice_info.height)
    x_scale = x_arr / float(slice_info.width)

    return (
        o[None, :] + (x_scale[:, None] * u[None, :]) + (y_scale[:, None] * v[None, :])
    )


# ---------------------------------------------------------------------------
# High-level: region area computation for a section
# ---------------------------------------------------------------------------


def get_region_areas(
    atlas_labels,
    seg_width: int,
    seg_height: int,
    slice_info,
    atlas_volume: np.ndarray,
    hemi_mask: Optional[np.ndarray],
    deform_map: Optional[Tuple] = None,
    precomputed_atlas_slice: Optional[np.ndarray] = None,
) -> Tuple[Any, np.ndarray]:
    """Build the atlas map for a slice and compute region areas.

    This performs atlas slice extraction from the annotation volume, applies
    non-linear warping if requested, and converts the resulting label map into
    a region-area dataframe.

    Parameters
    ----------
    atlas_labels : pd.DataFrame
        Atlas label definitions.
    seg_width, seg_height : int
        Segmentation image dimensions.
    slice_info : SliceInfo
        Registration context containing anchoring, dimensions, deformation,
        and damage mask.
    atlas_volume : np.ndarray
        3D atlas annotation volume.
    hemi_mask : np.ndarray or None
        Hemisphere mask volume.
    deform_map : tuple or None
        Precomputed deformation map to avoid redundant deformation calls.
    precomputed_atlas_slice : ndarray or None
        Already-extracted 2D atlas slice to skip slice extraction.

    Returns
    -------
    region_areas : pd.DataFrame
        DataFrame with region area statistics.
    atlas_map : np.ndarray
        2D atlas map for the section.
    """
    anchoring = slice_info.anchoring
    reg_width, reg_height = slice_info.width, slice_info.height
    deformation = slice_info.deformation
    damage_mask = slice_info.damage_mask

    atlas_map = load_atlas_image(
        anchoring,
        atlas_volume,
        deformation,
        (reg_width, reg_height),
        deform_map=deform_map,
        precomputed_slice=precomputed_atlas_slice,
    )
    region_areas = flat_to_dataframe(
        atlas_map, damage_mask, hemi_mask, (seg_width, seg_height)
    )
    return region_areas, atlas_map

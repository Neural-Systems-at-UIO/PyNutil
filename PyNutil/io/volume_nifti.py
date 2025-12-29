from __future__ import annotations

from typing import Optional

import numpy as np

from .nifti_writer import write_nifti


def scale_to_uint8(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    vmin = float(np.min(arr[finite]))
    vmax = float(np.max(arr[finite]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - vmin) * (255.0 / (vmax - vmin))
    scaled = np.clip(scaled, 0.0, 255.0)

    out = np.zeros(arr.shape, dtype=np.uint8)
    out[finite] = scaled[finite].round().astype(np.uint8)
    return out


def isotropic_resolution_um_for_volume(
    *,
    atlas_volume: Optional[np.ndarray],
    volume: np.ndarray,
    base_voxel_um: float,
    logger=None,
) -> float:
    if atlas_volume is None:
        return float(base_voxel_um)

    atlas_shape = np.array(atlas_volume.shape, dtype=np.float32)
    vol_shape = np.array(volume.shape, dtype=np.float32)
    if atlas_shape.shape != (3,) or vol_shape.shape != (3,) or np.any(vol_shape <= 0):
        return float(base_voxel_um)

    implied = atlas_shape / vol_shape
    iso_scale = float(np.median(implied))

    if logger is not None and (np.max(implied) - np.min(implied) > 1e-3):
        logger.warning(
            "Non-uniform volume scaling detected (atlas_shape=%s, volume_shape=%s). "
            "Writing isotropic voxel spacing using median scale %.6f.",
            tuple(int(x) for x in atlas_shape),
            tuple(int(x) for x in vol_shape),
            iso_scale,
        )

    return float(base_voxel_um * iso_scale)


def save_volume_niftis(
    *,
    output_folder: str,
    interpolated_volume: Optional[np.ndarray],
    frequency_volume: Optional[np.ndarray],
    damage_volume: Optional[np.ndarray] = None,
    atlas_volume: Optional[np.ndarray],
    voxel_size_um: Optional[float],
    logger=None,
) -> None:
    if interpolated_volume is None and frequency_volume is None:
        return

    base_voxel_um = float(voxel_size_um) if voxel_size_um is not None else 1.0

    def _save_one(volume: np.ndarray, *, name: str) -> None:
        vol_u8 = scale_to_uint8(volume)
        res = isotropic_resolution_um_for_volume(
            atlas_volume=atlas_volume,
            volume=vol_u8,
            base_voxel_um=base_voxel_um,
            logger=logger,
        )
        write_nifti(vol_u8, res, f"{output_folder}/interpolated_volume/{name}")

    if interpolated_volume is not None:
        _save_one(interpolated_volume, name="interpolated_volume")

    if frequency_volume is not None:
        _save_one(frequency_volume, name="frequency_volume")

    if damage_volume is not None:
        _save_one(damage_volume, name="damage_volume")


# Backwards-compatible alias (internal name used by earlier refactor iterations).
save_interpolated_volumes = save_volume_niftis

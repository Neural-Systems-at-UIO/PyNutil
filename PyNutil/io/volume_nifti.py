from __future__ import annotations

from typing import Optional

import numpy as np

from .nifti_writer import write_nifti
from ..results.volume import VolumeResult
from .atlas_loader import resolve_atlas


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


def save_volumes(
    *,
    output_folder: str,
    volumes: VolumeResult,
    atlas: object,
    logger=None,
) -> None:
    """Save atlas-space volumes as NIfTI files.

    Parameters
    ----------
    output_folder
        Base output directory where the ``interpolated_volume`` subdirectory
        will be created.
    volumes
        :class:`~PyNutil.VolumeResult` returned by
        :func:`~PyNutil.interpolate_volume`.
    atlas
        Atlas definition used to infer isotropic voxel spacing. Accepts a
        BrainGlobe atlas object or :class:`~PyNutil.AtlasData`.
    logger
        Optional logger used to report non-uniform output scaling.

    Notes
    -----
    Each written volume is scaled to 8-bit before export. Output files are
    written into ``<output_folder>/interpolated_volume``.

    Examples
    --------
    Save the volumes returned by :func:`PyNutil.interpolate_volume`:

    >>> image_series = read_segmentation_dir("path/to/segmentations/", pixel_id=[0, 0, 0])
    >>> registration = read_alignment("path/to/alignment.json")
    >>> volumes = interpolate_volume(
    ...     image_series=image_series,
    ...     registration=registration,
    ...     atlas=atlas,
    ... )
    >>> save_volumes(
    ...     output_folder="path/to/output",
    ...     volumes=volumes,
    ...     atlas=atlas,
    ... )
    """
    resolved = resolve_atlas(atlas)
    base_voxel_um = float(resolved.voxel_size_um) if resolved.voxel_size_um is not None else 1.0

    def _save_one(volume: np.ndarray, *, name: str) -> None:
        vol_u8 = scale_to_uint8(volume)
        res = isotropic_resolution_um_for_volume(
            atlas_volume=resolved.volume,
            volume=vol_u8,
            base_voxel_um=base_voxel_um,
            logger=logger,
        )
        write_nifti(vol_u8, res, f"{output_folder}/interpolated_volume/{name}")

    _save_one(volumes.value, name="interpolated_volume")
    _save_one(volumes.frequency, name="frequency_volume")
    _save_one(volumes.damage, name="damage_volume")

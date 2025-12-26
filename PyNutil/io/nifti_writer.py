from __future__ import annotations

import os
from typing import Optional

import numpy as np


def write_nifti(
    volume: np.ndarray,
    voxel_size_um: float,
    output_path: str,
    origin_offsets_um: Optional[np.ndarray] = None,
) -> None:
    """Write a NIfTI volume with a microns-based affine.

    The header is written with both qform and sform set and units set to microns.

    Args:
        volume: 3D volume array. Saved as uint8.
        voxel_size_um: Isotropic voxel size in microns.
        output_path: Output path without extension; ".nii.gz" is appended.
        origin_offsets_um: Optional XYZ translation offsets in microns.
    """

    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("nibabel is required for write_nifti") from exc

    if origin_offsets_um is None:
        origin_offsets_um = np.array([0, 0, 0], dtype=np.float32)

    dims = np.array(volume.shape, dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] *= float(voxel_size_um)
    affine[:3, 3] = -0.5 * dims * float(voxel_size_um) + np.asarray(
        origin_offsets_um, dtype=np.float32
    )

    img = nib.Nifti1Image(np.asarray(volume, dtype=np.uint8), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header["xyzt_units"] = 3

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path + ".nii.gz")

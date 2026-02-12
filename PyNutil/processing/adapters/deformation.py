"""Deformation providers for non-linear registration.

Deformation providers add non-linear warping to registration data.
They take RegistrationData with linear anchoring and add deformation
functions to each slice.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from scipy import ndimage
import numpy as np

from .base import DeformationFunction, DeformationProvider, RegistrationData


class VisuAlignDeformationProvider(DeformationProvider):
    """Adds VisuAlign non-linear deformation from markers.

    Can load from:
    - Same JSON file as anchoring (if markers are embedded)
    - Separate VisuAlign JSON file
    """

    name: str = "visualign"

    def __init__(self, path: Optional[str] = None):
        """Initialize provider.

        Args:
            path: Optional separate file with VisuAlign markers.
                  If None, uses markers from the anchoring file.
        """
        self.path = path

    def _load_markers(self, path: str) -> Dict[int, List[List[float]]]:
        """Load markers from a VisuAlign JSON file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        markers = {}
        for s in data.get("slices", []):
            nr = s.get("nr", 0)
            if "markers" in s and s["markers"]:
                markers[nr] = s["markers"]

        return markers

    def _create_deformation(
        self, width: int, height: int, markers: List[List[float]]
    ) -> Tuple[DeformationFunction, DeformationFunction]:
        """Create deformation functions from VisuAlign markers.

        Returns:
            Tuple of (inverse_deform, forward_deform) functions.
            - inverse_deform: maps from deformed to original (transform_vec)
            - forward_deform: maps from original to deformed (forwardtransform_vec)
        """
        from .visualign_deformations import (
            triangulate,
            transform_vec,
            forwardtransform_vec,
        )

        triangulation = triangulate(width, height, markers)

        def deform_inverse(
            x: np.ndarray, y: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            return transform_vec(triangulation, x, y)

        def deform_forward(
            x: np.ndarray, y: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            return forwardtransform_vec(triangulation, x, y)

        return deform_inverse, deform_forward

    def apply(self, data: RegistrationData) -> RegistrationData:
        """Add VisuAlign deformation to slices."""
        # Load markers from separate file if specified
        if self.path:
            markers_by_nr = self._load_markers(self.path)
        else:
            # Use markers from raw slice data
            markers_by_nr = {}
            for s in data.slices:
                raw = s.metadata.get("_raw_slice", {})
                if "markers" in raw and raw["markers"]:
                    markers_by_nr[s.section_number] = raw["markers"]

        # Apply deformation to each slice
        for s in data.slices:
            markers = markers_by_nr.get(s.section_number)
            if markers:
                inverse_deform, forward_deform = self._create_deformation(
                    s.width, s.height, markers
                )
                s.deformation = inverse_deform
                s.forward_deformation = forward_deform
                s.metadata["markers"] = markers

        return data


class BrainGlobeDeformationProvider(DeformationProvider):
    """Adds deformation from brainglobe-registration displacement field TIFFs.

    Loads ``deformation_field_0.tiff`` (y-displacement) and
    ``deformation_field_1.tiff`` (x-displacement) from the registration
    output directory.  These displacement fields map from the brain section
    pixel space to the atlas slice pixel space::

        atlas_x = brain_x + field_1[brain_y, brain_x]
        atlas_y = brain_y + field_0[brain_y, brain_x]

    This provider builds both deformation directions:
    - ``deformation`` (inverse): deformed -> original
    - ``forward_deformation``: original -> deformed

    The TIFF displacement fields define one direction only. The opposite
    direction is approximated by inverting the displacement field via
    self-warp-and-negate (sample displacement at displaced coordinates and
    multiply by ``-1``).
    """

    name: str = "brainglobe"

    def __init__(self, reg_dir: Optional[str] = None):
        """Initialize provider.

        Args:
            reg_dir: Directory containing deformation field TIFFs.
                     If None, the directory is taken from SliceInfo metadata.
        """
        self.reg_dir = reg_dir

    def _load_fields(
        self, reg_dir: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load deformation field TIFFs from *reg_dir*."""
        import tifffile

        f0_path = os.path.join(reg_dir, "deformation_field_0.tiff")
        f1_path = os.path.join(reg_dir, "deformation_field_1.tiff")

        if not os.path.isfile(f0_path) or not os.path.isfile(f1_path):
            return None, None

        field_0 = tifffile.imread(f0_path).astype(np.float32)  # y-displacement
        field_1 = tifffile.imread(f1_path).astype(np.float32)  # x-displacement
        return field_0, field_1

    @staticmethod
    def _create_displacement_deformation(
        disp_x: np.ndarray,
        disp_y: np.ndarray,
    ) -> DeformationFunction:
        """Create a callable from displacement fields in atlas registration space."""
        field_h, field_w = disp_x.shape

        def deform(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            x_arr = np.asarray(x, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32)

            xi = np.clip(np.round(x_arr).astype(np.int32), 0, field_w - 1)
            yi = np.clip(np.round(y_arr).astype(np.int32), 0, field_h - 1)
            return x_arr + disp_x[yi, xi], y_arr + disp_y[yi, xi]

        return deform

    @staticmethod
    def _invert_displacement_field(
        disp_x: np.ndarray,
        disp_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate inverse displacement via nearest-neighbor fill."""
        field_h, field_w = disp_x.shape
        yy, xx = np.indices((field_h, field_w), dtype=np.intp)

        # Initialize inverse fields with NaNs
        inv_x = np.full_like(disp_x, np.nan, dtype=np.float32)
        inv_y = np.full_like(disp_y, np.nan, dtype=np.float32)

        # Populate known samples
        inv_x[yy, xx] = disp_x[yy, xx]
        inv_y[yy, xx] = disp_y[yy, xx]

        # Nearest-neighbor NaN fill via distance transform
        mask = np.isnan(inv_x)

        indices = ndimage.distance_transform_edt(
            mask,
            return_distances=False,
            return_indices=True,
        )

        inv_x_filled = inv_x[tuple(indices)]
        inv_y_filled = inv_y[tuple(indices)]

        return (
            inv_x_filled.astype(np.float32, copy=False),
            inv_y_filled.astype(np.float32, copy=False),
        )
    @staticmethod
    def _create_deformation(
        field_0: np.ndarray,
        field_1: np.ndarray,
        atlas_w: int,
        atlas_h: int,
    ) -> Tuple[DeformationFunction, DeformationFunction]:
        """Create inverse + forward deformation functions from displacement fields."""
        brain_h, brain_w = field_0.shape

        def deform_inverse(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Scale from reg / atlas-slice space to brain-section pixel space
            x_brain = x * (brain_w / atlas_w)
            y_brain = y * (brain_h / atlas_h)

            xi = np.clip(np.round(x_brain).astype(np.int32), 0, brain_w - 1)
            yi = np.clip(np.round(y_brain).astype(np.int32), 0, brain_h - 1)

            dx = field_1[yi, xi]
            dy = field_0[yi, xi]

            return x_brain + dx, y_brain + dy

        # Build dense displacement in atlas registration space from the
        # provided (inverse) deformation, then invert that displacement.
        yy, xx = np.indices((atlas_h, atlas_w), dtype=np.float32)
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        warped_x, warped_y = deform_inverse(flat_x, flat_y)
        inverse_disp_x = (warped_x - flat_x).reshape(atlas_h, atlas_w).astype(np.float32)
        inverse_disp_y = (warped_y - flat_y).reshape(atlas_h, atlas_w).astype(np.float32)
        forward_disp_x, forward_disp_y = (
            BrainGlobeDeformationProvider._invert_displacement_field(
                inverse_disp_x, inverse_disp_y
            )
        )
        deform_forward = BrainGlobeDeformationProvider._create_displacement_deformation(
            forward_disp_x, forward_disp_y
        )

        return deform_inverse, deform_forward

    def apply(self, data: RegistrationData) -> RegistrationData:
        """Add brainglobe deformation to slices."""
        for s in data.slices:
            if s.metadata.get("registration_type") != "brainglobe":
                continue

            reg_dir = self.reg_dir or s.metadata.get("registration_dir")
            if not reg_dir:
                continue

            field_0, field_1 = self._load_fields(reg_dir)
            if field_0 is None:
                continue

            inverse_deform, forward_deform = self._create_deformation(
                field_0, field_1, s.width, s.height
            )
            s.deformation = inverse_deform
            s.forward_deformation = forward_deform
            s.metadata["deformation_type"] = "brainglobe"

        return data

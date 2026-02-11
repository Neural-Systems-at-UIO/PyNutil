"""Deformation providers for non-linear registration.

Deformation providers add non-linear warping to registration data.
They take RegistrationData with linear anchoring and add deformation
functions to each slice.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class DisplacementFieldProvider(DeformationProvider):
    """Adds deformation from a displacement field.

    This allows using custom non-linear registration tools that output
    displacement fields instead of control points.

    The displacement field should be a 3D numpy array of shape (H, W, 2)
    where [:,:,0] is the x-displacement and [:,:,1] is the y-displacement.
    """

    name: str = "displacement_field"

    def __init__(
        self,
        field_path: Optional[str] = None,
        field_loader: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ):
        """Initialize provider.

        Args:
            field_path: Path to a .npy file containing displacement fields.
                        Should be a dict mapping section_number -> field array.
            field_loader: Callable that takes section_number and returns the
                         displacement field array (or None if not available).
        """
        self.field_path = field_path
        self.field_loader = field_loader
        self._fields: Optional[Dict[int, np.ndarray]] = None

    def _load_fields(self) -> Dict[int, np.ndarray]:
        """Load displacement fields from file."""
        if self._fields is not None:
            return self._fields

        if self.field_path:
            self._fields = np.load(self.field_path, allow_pickle=True).item()
        else:
            self._fields = {}

        return self._fields

    def _create_deformation(self, field: np.ndarray) -> DeformationFunction:
        """Create deformation function from displacement field."""

        def deform(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Sample displacement field at the given coordinates
            h, w = field.shape[:2]

            # Clamp coordinates to field bounds
            xi = np.clip(x.astype(np.int32), 0, w - 1)
            yi = np.clip(y.astype(np.int32), 0, h - 1)

            # Get displacements
            dx = field[yi, xi, 0]
            dy = field[yi, xi, 1]

            return x + dx, y + dy

        return deform

    def apply(self, data: RegistrationData) -> RegistrationData:
        """Add displacement field deformation to slices."""
        fields = self._load_fields() if self.field_path else {}

        for s in data.slices:
            field = None

            if self.field_loader:
                field = self.field_loader(s.section_number)
            elif s.section_number in fields:
                field = fields[s.section_number]

            if field is not None:
                s.deformation = self._create_deformation(field)
                s.metadata["deformation_type"] = "displacement_field"

        return data


class BrainGlobeDeformationProvider(DeformationProvider):
    """Adds deformation from brainglobe-registration displacement field TIFFs.

    Loads ``deformation_field_0.tiff`` (y-displacement) and
    ``deformation_field_1.tiff`` (x-displacement) from the registration
    output directory.  These displacement fields map from the brain section
    pixel space to the atlas slice pixel space::

        atlas_x = brain_x + field_1[brain_y, brain_x]
        atlas_y = brain_y + field_0[brain_y, brain_x]

    The deformation function accepts coordinates in the registration space
    (which equals the atlas-slice dimensions from ``|U|`` and ``|V|``),
    internally scales them to brain-section pixel space for the field lookup,
    and returns atlas-slice coordinates.
    """

    name: str = "brainglobe"

    def __init__(self, reg_dir: Optional[str] = None):
        """Initialize provider.

        Args:
            reg_dir: Directory containing deformation field TIFFs.
                     If None, the directory is taken from SliceInfo metadata.
        """
        self.reg_dir = reg_dir

    def _load_fields(self, reg_dir: str) -> Tuple[np.ndarray, np.ndarray]:
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
    def _create_deformation(
        field_0: np.ndarray,
        field_1: np.ndarray,
        atlas_w: int,
        atlas_h: int,
    ) -> DeformationFunction:
        """Create a deformation function from displacement fields.

        The returned function maps registration-space (atlas-slice) coordinates
        to atlas-slice coordinates via the brain-section displacement lookup.
        """
        brain_h, brain_w = field_0.shape

        def deform(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Scale from reg / atlas-slice space to brain-section pixel space
            x_brain = x * (brain_w / atlas_w)
            y_brain = y * (brain_h / atlas_h)

            xi = np.clip(np.round(x_brain).astype(np.int32), 0, brain_w - 1)
            yi = np.clip(np.round(y_brain).astype(np.int32), 0, brain_h - 1)

            dx = field_1[yi, xi]
            dy = field_0[yi, xi]

            return x_brain + dx, y_brain + dy

        return deform

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

            s.deformation = self._create_deformation(
                field_0, field_1, s.width, s.height
            )
            s.metadata["deformation_type"] = "brainglobe"

        return data

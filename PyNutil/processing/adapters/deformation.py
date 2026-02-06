"""Deformation providers for non-linear registration.

Deformation providers add non-linear warping to registration data.
They take RegistrationData with linear anchoring and add deformation
functions to each slice.
"""

from __future__ import annotations

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
        from .visualign_deformations import triangulate, transform_vec, forwardtransform_vec

        triangulation = triangulate(width, height, markers)

        def deform_inverse(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return transform_vec(triangulation, x, y)

        def deform_forward(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
                inverse_deform, forward_deform = self._create_deformation(s.width, s.height, markers)
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

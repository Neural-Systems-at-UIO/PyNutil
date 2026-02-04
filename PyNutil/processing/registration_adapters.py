"""Modular registration system with composable components.

This module provides a flexible, mix-and-match architecture for registration:
- **Anchoring Loaders**: Load basic slice info and linear registration
- **Deformation Providers**: Add non-linear warping
- **Damage Providers**: Add damage/exclusion masks

Example workflows:
    # Standard QUINT workflow (QuickNII + VisuAlign + QCAlign all in one file)
    data = load_registration("alignment.json")

    # QuickNII linear only, no deformation
    data = load_registration("quicknii.json", apply_deformation=False)

    # QuickNII anchoring + custom deformation field
    data = load_registration(
        "quicknii.json",
        deformation_provider=DisplacementFieldProvider("deformation.npy")
    )

    # ABBA anchoring + QCAlign damage from separate file
    data = load_registration(
        "abba_export.json",
        damage_provider=QCAlignDamageProvider("qcalign.json")
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np


# =============================================================================
# Type Aliases
# =============================================================================

# Takes (x_coords, y_coords) arrays and returns (x_warped, y_warped)
DeformationFunction = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SliceInfo:
    """Standardized slice information from any registration source.

    This is the common data structure that all components produce and consume.
    Components can add or modify fields as data flows through the pipeline.
    """

    # Section identifier (filename or number)
    section_id: str
    section_number: int

    # Registration dimensions (in registration space)
    width: int
    height: int

    # Anchoring: 9 values defining the 3D plane in atlas space
    # [ox, oy, oz, ux, uy, uz, vx, vy, vz]
    # o = origin, u = horizontal vector, v = vertical vector
    anchoring: List[float]

    # Non-linear deformation function (optional)
    # Takes (x, y) coordinate arrays and returns warped (x', y') arrays
    # This is the INVERSE direction: maps from deformed space back to original
    # Used by quantification (transform_vec)
    deformation: Optional[DeformationFunction] = None

    # Forward deformation function (optional)
    # Maps from original space to deformed space
    # Used by volume projection (forwardtransform_vec)
    forward_deformation: Optional[DeformationFunction] = None

    # Damage/exclusion mask (optional)
    # 2D boolean array where True = damaged/excluded
    damage_mask: Optional[np.ndarray] = None

    # Additional metadata from various sources
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationData:
    """Complete registration data from an alignment file."""

    slices: List[SliceInfo]
    grid_spacing: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_slice_by_number(self, number: int) -> Optional[SliceInfo]:
        """Find a slice by its section number."""
        for s in self.slices:
            if s.section_number == number:
                return s
        return None


# =============================================================================
# Abstract Base Classes
# =============================================================================

class AnchoringLoader(ABC):
    """Loads basic slice info and linear registration (anchoring).

    This is the foundation - it provides section identification,
    dimensions, and the linear transformation to atlas space.

    Examples: QuickNII, DeepSlice, ABBA
    """

    name: str = "base"
    file_extensions: List[str] = []

    @abstractmethod
    def load(self, path: str) -> RegistrationData:
        """Load registration data from a file.

        Returns RegistrationData with slices containing:
        - section_id, section_number
        - width, height
        - anchoring

        Deformation and damage_mask should be None (added by providers).
        """
        pass

    @classmethod
    def can_handle(cls, path: str) -> bool:
        """Check if this loader can handle the given file."""
        ext = Path(path).suffix.lower()
        return ext in cls.file_extensions


class DeformationProvider(ABC):
    """Adds non-linear deformation to slice info.

    Takes RegistrationData and adds deformation functions to each slice.

    Examples: VisuAlign markers, displacement fields, B-spline transforms
    """

    name: str = "base"

    @abstractmethod
    def apply(self, data: RegistrationData) -> RegistrationData:
        """Add deformation functions to all slices.

        Args:
            data: RegistrationData with basic anchoring.

        Returns:
            RegistrationData with deformation functions added.
        """
        pass


class DamageProvider(ABC):
    """Adds damage/exclusion masks to slice info.

    Takes RegistrationData and adds damage masks to each slice.

    Examples: QCAlign grid, manual ROI masks, tissue detection
    """

    name: str = "base"

    @abstractmethod
    def apply(self, data: RegistrationData) -> RegistrationData:
        """Add damage masks to all slices.

        Args:
            data: RegistrationData (may already have deformation).

        Returns:
            RegistrationData with damage masks added.
        """
        pass


# =============================================================================
# QUINT Workflow Components
# =============================================================================

class QuintAnchoringLoader(AnchoringLoader):
    """Loads anchoring from QUINT JSON files (QuickNII, DeepSlice, VisuAlign).

    This loader extracts only the linear registration (anchoring).
    Non-linear deformation and damage are handled by separate providers.
    """

    name: str = "quint"
    file_extensions: List[str] = [".json"]

    def load(self, path: str) -> RegistrationData:
        """Load anchoring from a QUINT JSON file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        slices = []
        for s in data.get("slices", []):
            anchoring = s.get("anchoring", [])
            if not anchoring:
                continue

            slices.append(SliceInfo(
                section_id=s.get("filename", str(s.get("nr", 0))),
                section_number=s.get("nr", 0),
                width=s.get("width", 0),
                height=s.get("height", 0),
                anchoring=anchoring,
                deformation=None,  # Added by DeformationProvider
                damage_mask=None,  # Added by DamageProvider
                metadata={
                    "filename": s.get("filename"),
                    # Store raw data for providers to use
                    "_raw_slice": s,
                },
            ))

        return RegistrationData(
            slices=slices,
            grid_spacing=data.get("gridspacing"),
            metadata={
                "target": data.get("target"),
                "target-resolution": data.get("target-resolution"),
                "propagate": data.get("propagate", False),
                "_raw_data": data,  # Keep for providers
            },
        )


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
        self._markers_by_nr: Optional[Dict[int, List[List[float]]]] = None

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


class QCAlignDamageProvider(DamageProvider):
    """Adds QCAlign damage masks from grid data.

    Can load from:
    - Same JSON file as anchoring (if grid is embedded)
    - Separate QCAlign JSON file
    """

    name: str = "qcalign"

    def __init__(self, path: Optional[str] = None):
        """Initialize provider.

        Args:
            path: Optional separate file with QCAlign damage grids.
                  If None, uses grids from the anchoring file.
        """
        self.path = path

    def _load_grids(self, path: str) -> Tuple[Dict[int, Dict[str, Any]], Optional[int]]:
        """Load grid data from a QCAlign JSON file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        grids = {}
        for s in data.get("slices", []):
            nr = s.get("nr", 0)
            if "grid" in s and s["grid"]:
                grids[nr] = s

        return grids, data.get("gridspacing")

    def _create_damage_mask(
        self, slice_data: Dict[str, Any], grid_spacing: int
    ) -> np.ndarray:
        """Create damage mask from QCAlign grid.

        Returns a mask at grid resolution where 1=undamaged, 0=damaged.
        Downstream code is responsible for resizing to the needed resolution.
        This matches the original behavior where the grid-sized mask was passed
        to flat_to_dataframe which resized it.
        """
        from .utils import create_damage_mask

        # create_damage_mask returns 1=undamaged, 0=damaged at grid resolution
        binary_mask = create_damage_mask(slice_data, grid_spacing)

        return binary_mask

    def apply(self, data: RegistrationData) -> RegistrationData:
        """Add QCAlign damage masks to slices."""
        # Load grids from separate file if specified
        if self.path:
            grids_by_nr, grid_spacing = self._load_grids(self.path)
        else:
            # Use grids from raw slice data
            grids_by_nr = {}
            grid_spacing = data.grid_spacing
            for s in data.slices:
                raw = s.metadata.get("_raw_slice", {})
                if "grid" in raw and raw["grid"]:
                    grids_by_nr[s.section_number] = raw

        if not grid_spacing:
            return data  # Can't create masks without grid spacing

        # Apply damage mask to each slice
        for s in data.slices:
            grid_data = grids_by_nr.get(s.section_number)
            if grid_data:
                s.damage_mask = self._create_damage_mask(grid_data, grid_spacing)
                s.metadata["grid"] = grid_data.get("grid")

        return data


# =============================================================================
# Displacement Field Deformation (for custom non-linear tools)
# =============================================================================

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


# =============================================================================
# Registry and Loading
# =============================================================================

class AnchoringLoaderRegistry:
    """Registry for anchoring loaders."""

    _loaders: Dict[str, Type[AnchoringLoader]] = {}

    @classmethod
    def register(cls, loader_class: Type[AnchoringLoader]) -> None:
        cls._loaders[loader_class.name] = loader_class

    @classmethod
    def get(cls, name: str) -> AnchoringLoader:
        if name not in cls._loaders:
            available = ", ".join(cls._loaders.keys())
            raise ValueError(f"Unknown anchoring loader '{name}'. Available: {available}")
        return cls._loaders[name]()

    @classmethod
    def detect(cls, path: str) -> Optional[AnchoringLoader]:
        for loader_class in cls._loaders.values():
            if loader_class.can_handle(path):
                return loader_class()
        return None


# Register built-in loaders
AnchoringLoaderRegistry.register(QuintAnchoringLoader)


def load_registration(
    path: str,
    loader_name: Optional[str] = None,
    apply_deformation: bool = True,
    apply_damage: bool = True,
    deformation_provider: Optional[DeformationProvider] = None,
    damage_provider: Optional[DamageProvider] = None,
) -> RegistrationData:
    """Load registration data with composable pipeline.

    This is the main entry point for loading registration data. It supports
    mixing and matching different components.

    Args:
        path: Path to the registration file.
        loader_name: Explicit loader name, or None for auto-detection.
        apply_deformation: Whether to apply deformation from the file.
                          Set False to use only linear anchoring.
        apply_damage: Whether to apply damage masks from the file.
        deformation_provider: Custom deformation provider to use instead of
                             the default (VisuAlign for QUINT files).
        damage_provider: Custom damage provider to use instead of
                        the default (QCAlign for QUINT files).

    Returns:
        RegistrationData with all components applied.

    Examples:
        # Standard QUINT workflow
        data = load_registration("alignment.json")

        # Linear only (no VisuAlign deformation)
        data = load_registration("alignment.json", apply_deformation=False)

        # QuickNII with custom displacement field
        data = load_registration(
            "quicknii.json",
            deformation_provider=DisplacementFieldProvider("my_field.npy")
        )

        # Separate anchoring and damage files
        data = load_registration(
            "quicknii.json",
            damage_provider=QCAlignDamageProvider("qcalign_output.json")
        )
    """
    # 1. Load anchoring
    if loader_name:
        loader = AnchoringLoaderRegistry.get(loader_name)
    else:
        loader = AnchoringLoaderRegistry.detect(path)
        if loader is None:
            raise ValueError(f"Could not detect loader for '{path}'")

    data = loader.load(path)

    # 2. Apply deformation
    if apply_deformation:
        if deformation_provider:
            data = deformation_provider.apply(data)
        else:
            # Default: VisuAlign for QUINT files
            data = VisuAlignDeformationProvider().apply(data)

    # 3. Apply damage
    if apply_damage:
        if damage_provider:
            data = damage_provider.apply(data)
        else:
            # Default: QCAlign for QUINT files
            data = QCAlignDamageProvider().apply(data)

    return data


# =============================================================================
# Legacy Compatibility
# =============================================================================

# These maintain backwards compatibility with code expecting the old API

class RegistrationAdapter(AnchoringLoader):
    """Alias for AnchoringLoader (backwards compatibility)."""
    pass


class RegistrationAdapterRegistry:
    """Backwards-compatible registry wrapping the new system."""

    _adapters = AnchoringLoaderRegistry._loaders

    @classmethod
    def register(cls, adapter_class: Type[AnchoringLoader]) -> None:
        AnchoringLoaderRegistry.register(adapter_class)

    @classmethod
    def get(cls, name: str) -> AnchoringLoader:
        return AnchoringLoaderRegistry.get(name)

    @classmethod
    def load(cls, path: str, format_name: Optional[str] = None) -> RegistrationData:
        return load_registration(path, loader_name=format_name)


# Backwards compatibility alias
QuintAdapter = QuintAnchoringLoader

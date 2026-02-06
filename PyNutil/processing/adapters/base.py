"""Base classes and data structures for the adapter system.

This module contains the core abstractions that all adapters share:
- SliceInfo: Standardized slice information
- RegistrationData: Complete registration data container
- Abstract base classes for Anchoring, Deformation, and Damage providers
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

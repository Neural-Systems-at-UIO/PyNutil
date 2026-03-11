"""Adapter system for registration and segmentation formats.

This package provides a modular, plugin-based architecture for supporting
different registration tools (QuickNII, VisuAlign, ABBA) and segmentation
formats (Cellpose, ilastik, etc.).

The registration system uses a composable pipeline:
- **Anchoring Loaders**: Load basic slice info and linear registration
- **Deformation Providers**: Add non-linear warping
- **Damage Providers**: Add damage/exclusion masks

Example:
    from PyNutil.processing.adapters import load_registration

    # Standard QUINT workflow
    data = load_registration("alignment.json")

    # Linear only (no deformation)
    data = load_registration("quicknii.json", apply_deformation=False)

    )
"""

from .base import SliceInfo, RegistrationData
from .registry import load_registration
from .segmentation import SegmentationAdapter, SegmentationAdapterRegistry

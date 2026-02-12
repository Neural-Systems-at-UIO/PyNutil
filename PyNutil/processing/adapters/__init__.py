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

# Base classes and data structures
from .base import (
    DeformationFunction,
    SliceInfo,
    RegistrationData,
    AnchoringLoader,
    DeformationProvider,
    DamageProvider,
)

# Anchoring loaders
from .anchoring import (
    QuintAnchoringLoader,
)

# Deformation providers
from .deformation import (
    VisuAlignDeformationProvider,
)

# Damage providers
from .damage import (
    QCAlignDamageProvider,
)

# Registry and loading
from .registry import (
    AnchoringLoaderRegistry,
    load_registration,
)

# Segmentation adapters
from .segmentation import (
    ObjectInfo,
    SegmentationAdapter,
    BinaryAdapter,
    CellposeAdapter,
    SegmentationAdapterRegistry,
)

# VisuAlign deformations (for advanced use)
from .visualign_deformations import (
    triangulate,
    transform_vec,
    forwardtransform_vec,
)

__all__ = [
    # Base
    "DeformationFunction",
    "SliceInfo",
    "RegistrationData",
    "AnchoringLoader",
    "DeformationProvider",
    "DamageProvider",
    # Anchoring
    "QuintAnchoringLoader",
    # Deformation
    "VisuAlignDeformationProvider",
    # Damage
    "QCAlignDamageProvider",
    # Registry
    "AnchoringLoaderRegistry",
    "load_registration",
    # Segmentation
    "ObjectInfo",
    "SegmentationAdapter",
    "BinaryAdapter",
    "CellposeAdapter",
    "SegmentationAdapterRegistry",
    # VisuAlign
    "triangulate",
    "transform_vec",
    "forwardtransform_vec",
]

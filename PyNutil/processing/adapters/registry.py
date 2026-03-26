"""Registry and loading functions for the registration system.

This module provides the main entry point for loading registration data
with a composable pipeline of anchoring loaders, deformation providers,
and damage providers.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base import (
    AnchoringLoader,
    DeformationProvider,
    DamageProvider,
    RegistrationData,
)
from .anchoring import QuintAnchoringLoader, BrainGlobeRegistrationLoader
from .deformation import VisuAlignDeformationProvider, BrainGlobeDeformationProvider
from .damage import QCAlignDamageProvider


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
            raise ValueError(
                f"Unknown anchoring loader '{name}'. Available: {available}"
            )
        return cls._loaders[name]()

    @classmethod
    def detect(cls, path: str) -> Optional[AnchoringLoader]:
        for loader_class in cls._loaders.values():
            if loader_class.can_handle(path):
                return loader_class()
        return None


# Register built-in loaders — BrainGlobe first because its can_handle
# inspects JSON content (more specific than QuickNII's extension-only check).
AnchoringLoaderRegistry.register(BrainGlobeRegistrationLoader)
AnchoringLoaderRegistry.register(QuintAnchoringLoader)


def read_alignment(
    path: str,
    loader_name: Optional[str] = None,
    apply_deformation: bool = True,
    apply_damage: bool = True,
    deformation_provider: Optional[DeformationProvider] = None,
    damage_provider: Optional[DamageProvider] = None,
) -> RegistrationData:
    """Load registration data for downstream PyNutil processing.

    Parameters
    ----------
    path
        Path to a registration file produced by a supported workflow such as
        QuickNII, VisuAlign, or BrainGlobe registration.
    loader_name
        Explicit loader name to use. If ``None``, PyNutil attempts to detect
        the appropriate loader automatically from the file.
    apply_deformation
        If ``True``, apply non-linear deformation when supported by the input
        registration source. Set to ``False`` to keep only the linear
        anchoring transform.
    apply_damage
        If ``True``, load and attach damage masks when available.
    deformation_provider
        Optional custom deformation provider to use instead of the default
        provider selected for the detected registration type.
    damage_provider
        Optional custom damage provider to use instead of the default QCAlign
        integration.

    Returns
    -------
    RegistrationData
        Registration metadata and per-section transforms used by the
        segmentation, intensity, and coordinate pipelines.

    Examples
    --------
    Load a standard QUINT alignment file:

    >>> registration = read_alignment("alignment.json")

    Load BrainGlobe registration output in the same way:

    >>> registration = read_alignment("brainglobe-registration.json")
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
        elif data.metadata.get("registration_type") == "brainglobe":
            data = BrainGlobeDeformationProvider().apply(data)
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

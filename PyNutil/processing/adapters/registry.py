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

        # Separate anchoring and damage files
        from .damage import QCAlignDamageProvider
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
        elif data.metadata.get("registration_type") == "brainglobe":
            print("brainglobe detected")
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

"""DEPRECATED: This module has been merged into transforms.py.

All functions have been moved to PyNutil.processing.transforms.
This module is kept for backwards compatibility and will be removed in a future version.
"""

from __future__ import annotations

import warnings

from .transforms import (
    transform_to_registration,
    transform_to_atlas_space,
    image_to_atlas_space,
    get_transformed_coordinates,
    transform_points_to_atlas_space,
)

warnings.warn(
    "PyNutil.processing.transformations is deprecated. "
    "Use PyNutil.processing.transforms instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "transform_to_registration",
    "transform_to_atlas_space",
    "image_to_atlas_space",
    "get_transformed_coordinates",
    "transform_points_to_atlas_space",
]

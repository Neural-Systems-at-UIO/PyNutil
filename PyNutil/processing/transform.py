"""DEPRECATED: This module has been merged into transforms.py.

All functions have been moved to PyNutil.processing.transforms.
This module is kept for backwards compatibility and will be removed in a future version.
"""

from __future__ import annotations

import warnings

from .transforms import get_triangulation, get_region_areas

warnings.warn(
    "PyNutil.processing.transform is deprecated. "
    "Use PyNutil.processing.transforms instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["get_triangulation", "get_region_areas"]

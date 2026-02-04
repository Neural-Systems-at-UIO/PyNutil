"""DEPRECATED: This module has been split into smaller focused modules.

Functions have been moved to:
- PyNutil.io.loaders: File loading functions
- PyNutil.io.meshview_writer: MeshView JSON writing
- PyNutil.io.colormap: Colormap utilities

This module re-exports all functions for backwards compatibility.
It will be removed in a future version.
"""

from __future__ import annotations

import warnings

# Re-export from loaders
from .loaders import (
    open_custom_region_file,
    read_flat_file,
    read_seg_file,
    load_segmentation,
    load_quint_json,
)

# Re-export from meshview_writer
from .meshview_writer import (
    create_region_dict,
    write_hemi_points_to_meshview,
    write_points_to_meshview,
)

# Re-export from colormap (private function, but keep for any external use)
from .colormap import get_colormap_color as _get_colormap_color

warnings.warn(
    "PyNutil.io.read_and_write is deprecated. "
    "Use PyNutil.io.loaders, PyNutil.io.meshview_writer, "
    "and PyNutil.io.colormap instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # loaders
    "open_custom_region_file",
    "read_flat_file",
    "read_seg_file",
    "load_segmentation",
    "load_quint_json",
    # meshview_writer
    "create_region_dict",
    "write_hemi_points_to_meshview",
    "write_points_to_meshview",
    # colormap
    "_get_colormap_color",
]

"""PyNutil I/O subpackage.

This package contains modules for reading and writing data:
- atlas_loader: Loading BrainGlobe and custom atlases
- file_operations: Saving analysis outputs
- loaders: File loading (custom regions, flat files, segmentations, JSON)
- meshview_writer: MeshView JSON output
- colormap: Colormap utilities
- section_visualisation: Creating section PNG visualizations
- volume_nifti: NIfTI volume output
- propagation: Slice anchoring interpolation
- reconstruct_dzi: DZI image reconstruction

Note: read_and_write is deprecated, use loaders/meshview_writer/colormap instead.
"""

from .atlas_loader import (
    load_atlas_data,
    load_atlas_labels,
    load_custom_atlas,
    process_atlas_volume,
)
from .file_operations import save_analysis_output
from .loaders import (
    load_quint_json,
    load_segmentation,
    open_custom_region_file,
    read_flat_file,
    read_seg_file,
)
from .meshview_writer import (
    create_region_dict,
    write_hemi_points_to_meshview,
    write_points_to_meshview,
)
from .colormap import get_colormap_color
from .volume_nifti import save_volume_niftis

__all__ = [
    # atlas_loader
    "load_atlas_data",
    "load_atlas_labels",
    "load_custom_atlas",
    "process_atlas_volume",
    # file_operations
    "save_analysis_output",
    # loaders
    "load_quint_json",
    "load_segmentation",
    "open_custom_region_file",
    "read_flat_file",
    "read_seg_file",
    # meshview_writer
    "create_region_dict",
    "write_hemi_points_to_meshview",
    "write_points_to_meshview",
    # colormap
    "get_colormap_color",
    # volume_nifti
    "save_volume_niftis",
]

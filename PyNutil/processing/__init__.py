"""PyNutil processing subpackage.

This package contains modules for data processing and analysis:

Subpackages
-----------
- analysis: Quantification, counting, and intensity aggregation
- pipeline: Batch processing, section transformation, connected components
- adapters: Plugin system for segmentation and registration formats

Top-level modules
-----------------
- transforms: Coordinate transformation functions
- utils: Shared utility functions
- section_volume: 3D volume projection/interpolation
- atlas_map: Atlas slice extraction and region-area computation
"""

from .pipeline import (
    folder_to_atlas_space,
    folder_to_atlas_space_intensity,
)
from .analysis import (
    apply_custom_regions,
    map_to_custom_regions,
    quantify_intensity,
    quantify_labeled_points,
)
from .transforms import (
    get_region_areas,
    get_transformed_coordinates,
    transform_points_to_atlas_space,
    transform_to_atlas_space,
    transform_to_registration,
)
from .adapters.visualign_deformations import (
    transform_vec,
    triangulate,
)
from .section_volume import project_sections_to_volume
from .adapters import (
    # Segmentation adapters
    SegmentationAdapter,
    SegmentationAdapterRegistry,
    BinaryAdapter,
    CellposeAdapter,
    # Core data classes
    RegistrationData,
    SliceInfo,
    DeformationFunction,
    # Abstract base classes
    AnchoringLoader,
    DeformationProvider,
    DamageProvider,
    # QUINT workflow components
    QuintAnchoringLoader,
    VisuAlignDeformationProvider,
    QCAlignDamageProvider,
    # Registry
    AnchoringLoaderRegistry,
    # Main entry point
    load_registration,
)

__all__ = [
    # coordinate_extraction
    "folder_to_atlas_space",
    "folder_to_atlas_space_intensity",
    # data_analysis
    "apply_custom_regions",
    "map_to_custom_regions",
    "quantify_intensity",
    "quantify_labeled_points",
    # transforms
    "get_region_areas",
    "get_transformed_coordinates",
    "transform_points_to_atlas_space",
    "transform_to_atlas_space",
    "transform_to_registration",
    # visualign_deformations
    "transform_vec",
    "triangulate",
    # section_volume
    "project_sections_to_volume",
    # segmentation_adapters
    "SegmentationAdapter",
    "SegmentationAdapterRegistry",
    "BinaryAdapter",
    "CellposeAdapter",
    # registration_adapters
    "RegistrationData",
    "SliceInfo",
    "load_registration",
]

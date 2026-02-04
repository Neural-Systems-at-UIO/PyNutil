"""PyNutil processing subpackage.

This package contains modules for data processing and analysis:
- coordinate_extraction: Facade for pixel/centroid extraction (re-exports from submodules)
- batch_processor: Folder-level batch processing with threading
- section_processor: Single section transformation to atlas space
- connected_components: Connected component analysis and region assignment
- segmentation_adapters: Plugin system for segmentation formats (Cellpose, ilastik, etc.)
- registration_adapters: Plugin system for registration formats (QUINT, ABBA, etc.)
- counting_and_load: Region counting and image loading
- data_analysis: Quantification and aggregation
- transforms: Coordinate transformation functions
- visualign_deformations: Non-linear deformation via triangulation
- utils: Utility functions
- section_volume: 3D volume projection/interpolation
- aggregator: Intensity aggregation per region
- generate_target_slice: Atlas slice extraction
- image_loaders: Pixel ID detection
"""

from .coordinate_extraction import (
    folder_to_atlas_space,
    folder_to_atlas_space_intensity,
)
from .data_analysis import (
    apply_custom_regions,
    map_to_custom_regions,
    quantify_intensity,
    quantify_labeled_points,
)
from .transforms import (
    get_region_areas,
    get_triangulation,
    get_transformed_coordinates,
    image_to_atlas_space,
    transform_points_to_atlas_space,
    transform_to_atlas_space,
    transform_to_registration,
)
from .visualign_deformations import (
    transform_vec,
    triangulate,
)
from .section_volume import project_sections_to_volume
from .segmentation_adapters import (
    SegmentationAdapter,
    SegmentationAdapterRegistry,
    BinaryAdapter,
    CellposeAdapter,
)
from .registration_adapters import (
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
    # Custom deformation support
    DisplacementFieldProvider,
    # Registry
    AnchoringLoaderRegistry,
    # Main entry point
    load_registration,
    # Legacy compatibility
    RegistrationAdapter,
    RegistrationAdapterRegistry,
    QuintAdapter,
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
    "get_triangulation",
    "get_transformed_coordinates",
    "image_to_atlas_space",
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
    "ColorMaskAdapter",
    "LabeledImageAdapter",
    "get_segmentation_adapter",
    # registration_adapters
    "RegistrationAdapter",
    "RegistrationAdapterRegistry",
    "RegistrationData",
    "SliceInfo",
    "QuintAdapter",
    "load_registration",
]

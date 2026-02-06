"""Section processing and batch orchestration subpackage.

Contains:
- batch_processor: Folder-level batch processing with threading.
- section_processor: Single section transformation to atlas space.
- connected_components: Connected component analysis and region assignment.
"""

from .batch_processor import (
    folder_to_atlas_space,
    folder_to_atlas_space_intensity,
)

__all__ = [
    "folder_to_atlas_space",
    "folder_to_atlas_space_intensity",
]

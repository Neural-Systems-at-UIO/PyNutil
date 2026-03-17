"""Section processing and batch orchestration subpackage.

Contains:
- batch_processor: Folder-level batch processing with threading.
- section_processor: Single section transformation to atlas space.
- connected_components: Connected component analysis and region assignment.
"""

from .batch_processor import (
    seg_to_coords,
    image_to_coords,
)

__all__ = [
    "seg_to_coords",
    "image_to_coords",
]

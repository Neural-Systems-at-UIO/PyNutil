"""Quantification and aggregation subpackage.

Contains:
- data_analysis: Quantification and aggregation of labelled data.
- counting_and_load: Per-region pixel/object counting.
- aggregator: Intensity aggregation per atlas region.
"""

from .data_analysis import (
    apply_custom_regions,
    map_to_custom_regions,
    quantify_intensity,
    quantify_labeled_points,
)

__all__ = [
    "apply_custom_regions",
    "map_to_custom_regions",
    "quantify_intensity",
    "quantify_labeled_points",
]

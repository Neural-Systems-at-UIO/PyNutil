"""Damage providers for adding exclusion/damage masks.

Damage providers add damage/exclusion masks to registration data.
They take RegistrationData and add damage masks to each slice.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import DamageProvider, RegistrationData
from ...io.loaders import load_json_file


# ---------------------------------------------------------------------------
# Grid-based damage mask construction
# ---------------------------------------------------------------------------


def create_damage_mask(slice_info, section_grid, grid_spacing):
    """Create a binary damage mask from grid information in the given section.

    Args:
        section (dict): Dictionary with slice and grid data.
        grid_spacing (int): Space between grid marks.

    Returns:
        ndarray: Binary mask with damaged areas marked as 0.
    """
    width = slice_info.width
    height = slice_info.height
    grid_values = section_grid["grid"]
    gridx = section_grid["gridx"]
    gridy = section_grid["gridy"]

    ow, oh = slice_info.physical_dimensions
    xspacing = int(width * grid_spacing / ow)
    yspacing = int(height * grid_spacing / oh)
    x_coords = np.arange(gridx, width, xspacing)
    y_coords = np.arange(gridy, height, yspacing)

    num_markers = len(grid_values)
    markers = [
        (x_coords[i % len(x_coords)], y_coords[i // len(x_coords)])
        for i in range(num_markers)
    ]

    binary_image = np.ones((len(y_coords), len(x_coords)), dtype=int)

    for i, (x, y) in enumerate(markers):
        if grid_values[i] == 4:
            binary_image[y // yspacing, x // xspacing] = 0

    return binary_image


class QCAlignDamageProvider(DamageProvider):
    """Adds QCAlign damage masks from grid data.

    Can load from:
    - Same JSON file as anchoring (if grid is embedded)
    - Separate QCAlign JSON file
    """

    name: str = "qcalign"

    def __init__(self, path: Optional[str] = None):
        """Initialize provider.

        Args:
            path: Optional separate file with QCAlign damage grids.
                  If None, uses grids from the anchoring file.
        """
        self.path = path

    def _load_grids(self, path: str) -> Tuple[Dict[int, Dict[str, Any]], Optional[int]]:
        """Load grid data from a QCAlign JSON file."""
        data = load_json_file(path)

        grids = {}
        for s in data.get("slices", []):
            nr = s.get("nr", 0)
            if "grid" in s and s["grid"]:
                grids[nr] = s

        return grids, data.get("gridspacing")

    def apply(self, data: RegistrationData) -> RegistrationData:
        """Add QCAlign damage masks to slices."""
        # Load grids from separate file if specified
        if self.path:
            grids_by_nr, grid_spacing = self._load_grids(self.path)
        else:
            # Use grids from raw slice data
            grids_by_nr = {}
            grid_spacing = data.grid_spacing
            for s in data.slices:
                raw = s.metadata.get("_raw_slice", {})
                if "grid" in raw and raw["grid"]:
                    grids_by_nr[s.section_number] = raw

        if not grid_spacing:
            return data  # Can't create masks without grid spacing

        # Apply damage mask to each slice
        for s in data.slices:
            grid_data = grids_by_nr.get(s.section_number)
            if grid_data:
                s.damage_mask = create_damage_mask(s, grid_data, grid_spacing)
                s.metadata["grid"] = grid_data.get("grid")

        return data

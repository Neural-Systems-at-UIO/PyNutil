"""Damage providers for adding exclusion/damage masks.

Damage providers add damage/exclusion masks to registration data.
They take RegistrationData and add damage masks to each slice.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import DamageProvider, RegistrationData


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
        import json

        with open(path, "r") as f:
            data = json.load(f)

        grids = {}
        for s in data.get("slices", []):
            nr = s.get("nr", 0)
            if "grid" in s and s["grid"]:
                grids[nr] = s

        return grids, data.get("gridspacing")

    def _create_damage_mask(
        self, slice_data: Dict[str, Any], grid_spacing: int
    ) -> np.ndarray:
        """Create damage mask from QCAlign grid.

        Returns a mask at grid resolution where 1=undamaged, 0=damaged.
        Downstream code is responsible for resizing to the needed resolution.
        This matches the original behavior where the grid-sized mask was passed
        to flat_to_dataframe which resized it.
        """
        from ..utils import create_damage_mask

        # create_damage_mask returns 1=undamaged, 0=damaged at grid resolution
        binary_mask = create_damage_mask(slice_data, grid_spacing)

        return binary_mask

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
                s.damage_mask = self._create_damage_mask(grid_data, grid_spacing)
                s.metadata["grid"] = grid_data.get("grid")

        return data

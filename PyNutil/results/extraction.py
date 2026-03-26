from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..processing.reorientation import INTERNAL_ORIENTATION, reorient_points


@dataclass
class PointSetResult:
    """A reusable point-set payload shared across extraction workflows.

    Attributes:
        points: Atlas-space coordinates (N x 3).
        labels: Region labels aligned with ``points``.
        hemi_labels: Hemisphere labels aligned with ``points``.
        section_lengths: Per-section counts aligned with canonical arrays.
        point_values: Optional values aligned with ``points`` (e.g., intensity/RGB).
        undamaged_mask: Optional unfiltered undamaged mask.
        orientation: 3-letter BrainGlobe orientation string describing the
            coordinate system of ``points`` (e.g. "lpi", "asr").
        atlas_shape: Shape of the atlas volume in internal orientation,
            needed for reorienting points between coordinate systems.
    """

    points: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    hemi_labels: Optional[np.ndarray]
    section_lengths: List[int]
    point_values: Optional[np.ndarray] = None
    undamaged_mask: Optional[np.ndarray] = None
    orientation: str = "lpi"
    atlas_shape: Optional[Tuple[int, ...]] = None

    @staticmethod
    def _masked(arr: Optional[np.ndarray], mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return ``arr`` filtered by ``mask`` when both are present."""
        if arr is None or mask is None:
            return arr
        if len(arr) != len(mask):
            raise ValueError(
                f"Length mismatch between array ({len(arr)}) and mask ({len(mask)})."
            )
        return arr[mask]

    def _to_internal(self, pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Reorient points back to internal orientation if needed."""
        if pts is None or len(pts) == 0 or self.orientation == INTERNAL_ORIENTATION:
            return pts
        if self.atlas_shape is None:
            raise ValueError(
                "atlas_shape is required to reorient points back to internal orientation"
            )
        return reorient_points(pts, self.atlas_shape, INTERNAL_ORIENTATION,
                               source_orientation=self.orientation)

    def internal_points(self) -> Optional[np.ndarray]:
        """Return points reoriented to internal (lpi) orientation."""
        return self._to_internal(self.points)

    def filtered_points(self) -> Optional[np.ndarray]:
        """Return points filtered by undamaged mask when available."""
        return self._masked(self.points, self.undamaged_mask)

    def filtered_labels(self) -> Optional[np.ndarray]:
        """Return labels filtered by undamaged mask when available."""
        return self._masked(self.labels, self.undamaged_mask)

    def filtered_hemi_labels(self) -> Optional[np.ndarray]:
        """Return hemisphere labels filtered by undamaged mask when available."""
        return self._masked(self.hemi_labels, self.undamaged_mask)

    def filtered_point_values(self) -> Optional[np.ndarray]:
        """Return point values filtered by undamaged mask when available."""
        return self._masked(self.point_values, self.undamaged_mask)

    def filtered_section_lengths(self) -> List[int]:
        """Return per-section lengths after applying undamaged mask."""
        if self.undamaged_mask is None:
            return list(self.section_lengths)
        lengths: List[int] = []
        offset = 0
        for n in self.section_lengths:
            section_mask = self.undamaged_mask[offset : offset + n]
            lengths.append(int(np.count_nonzero(section_mask)))
            offset += n
        return lengths


@dataclass
class ExtractionResult:
    """User-facing extraction output with shared point-set structure."""

    points: PointSetResult
    objects: Optional[PointSetResult]
    section_filenames: List[str]
    region_areas: Optional[pd.DataFrame] = None
    region_intensities: Optional[pd.DataFrame] = None

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PerEntityArrays:
    """Concatenated per-entity arrays across all sections."""

    labels: np.ndarray
    hemi_labels: np.ndarray
    undamaged: np.ndarray
    section_lengths: List[int]

    def split(self):
        """Yield per-section slices as ``(labels, hemi, undamaged)`` tuples."""
        offset = 0
        for n in self.section_lengths:
            lab = self.labels[offset : offset + n]
            hemi = self.hemi_labels[offset : offset + n]
            und = self.undamaged[offset : offset + n]
            yield lab, hemi, und
            offset += n

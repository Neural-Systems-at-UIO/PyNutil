"""Anchoring loaders for loading linear registration data.

Anchoring loaders extract the basic slice information and linear
transformation (anchoring) from registration files.
"""

from __future__ import annotations

from typing import List

from .base import AnchoringLoader, RegistrationData, SliceInfo


class QuintAnchoringLoader(AnchoringLoader):
    """Loads anchoring from QUINT JSON files (QuickNII, DeepSlice, VisuAlign).

    This loader extracts only the linear registration (anchoring).
    Non-linear deformation and damage are handled by separate providers.
    """

    name: str = "quint"
    file_extensions: List[str] = [".json"]

    def load(self, path: str) -> RegistrationData:
        """Load anchoring from a QUINT JSON file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        slices = []
        for s in data.get("slices", []):
            anchoring = s.get("anchoring", [])
            if not anchoring:
                continue

            slices.append(SliceInfo(
                section_id=s.get("filename", str(s.get("nr", 0))),
                section_number=s.get("nr", 0),
                width=s.get("width", 0),
                height=s.get("height", 0),
                anchoring=anchoring,
                deformation=None,  # Added by DeformationProvider
                damage_mask=None,  # Added by DamageProvider
                metadata={
                    "filename": s.get("filename"),
                    # Store raw data for providers to use
                    "_raw_slice": s,
                },
            ))

        return RegistrationData(
            slices=slices,
            grid_spacing=data.get("gridspacing"),
            metadata={
                "target": data.get("target"),
                "target-resolution": data.get("target-resolution"),
                "propagate": data.get("propagate", False),
                "_raw_data": data,  # Keep for providers
            },
        )


# Backwards compatibility alias
QuintAdapter = QuintAnchoringLoader

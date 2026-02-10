"""Anchoring loaders for loading linear registration data.

Anchoring loaders extract the basic slice information and linear
transformation (anchoring) from registration files.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List

import numpy as np

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

            slices.append(
                SliceInfo(
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
                )
            )

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


class BrainGlobeRegistrationLoader(AnchoringLoader):
    """Load BrainGlobe registration output JSON files.

    BrainGlobe's registration output stores slice corners in physical units
    (microns) in TL/TR/BL/BR order. We convert those to atlas voxel space and
    emit a single SliceInfo with a QuickNII-compatible anchoring vector.
    """

    name: str = "brainglobe"
    file_extensions: List[str] = [".json"]

    @classmethod
    def can_handle(cls, path: str) -> bool:
        if not super().can_handle(path):
            return False

        import json

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            return False

        return "atlas_slice_corners" in data and "atlas" in data

    @staticmethod
    def _atlas_resolution_um(atlas_name: str) -> float:
        """Infer isotropic atlas resolution in microns from atlas name."""
        match = re.search(r"_(\d+(?:\.\d+)?)um$", atlas_name)
        if not match:
            raise ValueError(
                f"Could not infer atlas resolution from atlas name '{atlas_name}'"
            )
        return float(match.group(1))

    @staticmethod
    def _find_image_shape(registration_path: str) -> tuple[int, int]:
        """Find registration-space image dimensions from nearby TIFF outputs."""
        import tifffile

        folder = Path(registration_path).parent
        candidates = [
            folder / "downsampled.tiff",
            folder / "registered_atlas.tiff",
            folder / "registered_hemispheres.tiff",
        ]
        for candidate in candidates:
            if candidate.exists():
                image = tifffile.imread(candidate)
                if image.ndim < 2:
                    continue
                height, width = image.shape[:2]
                return int(height), int(width)

        raise FileNotFoundError(
            "Could not determine BrainGlobe registration dimensions. "
            "Expected one of downsampled.tiff, registered_atlas.tiff, "
            "or registered_hemispheres.tiff next to the registration JSON."
        )

    def load(self, path: str) -> RegistrationData:
        import json

        with open(path, "r") as f:
            data = json.load(f)

        corners_um = np.asarray(data["atlas_slice_corners"], dtype=np.float64)
        if corners_um.shape != (4, 3):
            raise ValueError(
                "Expected atlas_slice_corners to have shape (4, 3) in TL/TR/BL/BR order"
            )

        atlas_name = data["atlas"]
        resolution_um = self._atlas_resolution_um(atlas_name)
        corners_px = corners_um / resolution_um

        # BrainGlobe corner order: TL, TR, BL, BR
        top_left, top_right, bottom_left, _ = corners_px
        anchoring = [
            *top_left.tolist(),
            *(top_right - top_left).tolist(),
            *(bottom_left - top_left).tolist(),
        ]

        height, width = self._find_image_shape(path)

        section_number = int(data.get("atlas_2d_slice_index", 0))
        section_id = Path(path).stem
        slice_info = SliceInfo(
            section_id=section_id,
            section_number=section_number,
            width=width,
            height=height,
            anchoring=anchoring,
            deformation=None,
            damage_mask=None,
            metadata={
                "atlas": atlas_name,
                "atlas_resolution_um": resolution_um,
                "registration_dir": str(Path(path).parent),
                "_raw_data": data,
            },
        )

        return RegistrationData(
            slices=[slice_info],
            grid_spacing=None,
            metadata={
                "atlas": atlas_name,
                "source": "brainglobe-registration",
                "_raw_data": data,
            },
        )

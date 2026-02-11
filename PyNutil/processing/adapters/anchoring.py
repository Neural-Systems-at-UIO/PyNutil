"""Anchoring loaders for loading linear registration data.

Anchoring loaders extract the basic slice information and linear
transformation (anchoring) from registration files.
"""

from __future__ import annotations

import math
import os
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
    """Loads anchoring from brainglobe-registration output JSON.

    Converts ``atlas_slice_corners`` (in microns) to a QuickNII-compatible
    anchoring vector by:
    1. Converting microns to atlas voxels using the atlas resolution.
    2. Transforming from BrainGlobe orientation to PyNutil orientation
       (which applies ``transpose([2,0,1])[::-1,::-1,::-1]``).
    3. Computing O, U, V vectors from the TL, TR, BL corners.
    """

    name: str = "brainglobe"
    file_extensions: List[str] = [".json"]

    @classmethod
    def can_handle(cls, path: str) -> bool:
        """Detect brainglobe-registration JSON by looking for atlas_slice_corners."""
        import json

        if not path.endswith(".json"):
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return "atlas_slice_corners" in data
        except Exception:
            return False

    @staticmethod
    def _infer_resolution(atlas_name: str) -> float:
        """Extract voxel resolution in microns from the atlas name."""
        m = re.search(r"(\d+(?:\.\d+)?)um$", atlas_name)
        if not m:
            raise ValueError(
                f"Cannot infer resolution from atlas name '{atlas_name}'. "
                "Expected a name ending in '<number>um' (e.g. 'allen_mouse_25um')."
            )
        return float(m.group(1))

    @staticmethod
    def _get_atlas_shape(atlas_name: str):
        """Return the native (un-reoriented) atlas annotation shape."""
        import brainglobe_atlasapi

        bg = brainglobe_atlasapi.BrainGlobeAtlas(atlas_name=atlas_name)
        return bg.annotation.shape  # (AP, DV, LR)

    @staticmethod
    def _bg_to_pynutil(corners_vx: np.ndarray, bg_shape) -> np.ndarray:
        """Convert BrainGlobe voxel coords to PyNutil atlas coords.

        PyNutil reorients the atlas via ``transpose([2,0,1])[::-1,::-1,::-1]``.
        Given BrainGlobe coords ``(bg0, bg1, bg2)`` (AP, DV, LR) and the
        original annotation shape ``(S0, S1, S2)``, the PyNutil coords are:
            px = (S2 - 1) - bg2
            py = (S0 - 1) - bg0
            pz = (S1 - 1) - bg1
        """
        pn = np.zeros_like(corners_vx)
        pn[:, 0] = (bg_shape[2] - 1) - corners_vx[:, 2]
        pn[:, 1] = (bg_shape[0] - 1) - corners_vx[:, 0]
        pn[:, 2] = (bg_shape[1] - 1) - corners_vx[:, 1]
        return pn

    @staticmethod
    def _find_tiff_dims(reg_dir: str):
        """Read the dimensions of one of the output TIFFs in *reg_dir*."""
        import tifffile

        for name in ("downsampled.tiff", "registered_atlas.tiff", "registered_hemispheres.tiff"):
            p = os.path.join(reg_dir, name)
            if os.path.isfile(p):
                img = tifffile.imread(p)
                return img.shape[0], img.shape[1]  # (height, width)
        return None, None

    def load(self, path: str) -> RegistrationData:
        """Load anchoring from a brainglobe-registration JSON."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        atlas_name = data["atlas"]
        corners_um = np.array(data["atlas_slice_corners"], dtype=np.float64)
        resolution = self._infer_resolution(atlas_name)
        corners_vx = corners_um / resolution

        bg_shape = self._get_atlas_shape(atlas_name)
        pn_corners = self._bg_to_pynutil(corners_vx, bg_shape)

        # TL, TR, BL, BR
        O = pn_corners[0]
        U = pn_corners[1] - pn_corners[0]
        V = pn_corners[2] - pn_corners[0]
        anchoring = O.tolist() + U.tolist() + V.tolist()

        width = int(math.floor(math.hypot(*U))) + 1
        height = int(math.floor(math.hypot(*V))) + 1

        reg_dir = os.path.dirname(os.path.abspath(path))
        tiff_h, tiff_w = self._find_tiff_dims(reg_dir)

        section_number = data.get("atlas_2d_slice_index", 0)

        slices = [
            SliceInfo(
                section_id=str(section_number),
                section_number=section_number,
                width=width,
                height=height,
                anchoring=anchoring,
                deformation=None,
                damage_mask=None,
                metadata={
                    "registration_type": "brainglobe",
                    "registration_dir": reg_dir,
                    "atlas_name": atlas_name,
                    "resolution_um": resolution,
                    "bg_atlas_shape": bg_shape,
                    "tiff_width": tiff_w,
                    "tiff_height": tiff_h,
                },
            )
        ]

        return RegistrationData(
            slices=slices,
            metadata={
                "registration_type": "brainglobe",
                "atlas_name": atlas_name,
                "_raw_data": data,
            },
        )

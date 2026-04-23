"""Segmentation format adapters.

This module defines the interface for supporting different segmentation
tool outputs (Cellpose, ilastik, StarDist, custom masks, etc.).

To add support for a new segmentation format:
1. Create a new class that inherits from SegmentationAdapter
2. Implement the required abstract methods
3. Register it with the SegmentationAdapterRegistry
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import cv2
import numpy as np

from ...io.reconstruct_dzi import reconstruct_dzi


def _detect_pixel_id(segmentation: np.ndarray) -> np.ndarray:
    """Infer the foreground pixel id as the second most common value.

    This is robust to any background colour (black, white, or other)
    by finding the most common value (background) and returning the
    next most common value (foreground).
    """
    if segmentation.ndim == 2:
        flat = segmentation.ravel()
        values, counts = np.unique(flat, return_counts=True)
        if len(values) < 2:
            return np.asarray([255])
        # Second most common value is the foreground
        order = np.argsort(-counts)
        pixel_id = [int(values[order[1]])]
    else:
        # Treat each pixel as a tuple; find second most common colour
        h, w, c = segmentation.shape
        flat = segmentation.reshape(-1, c)
        # Use structured array for unique colour detection
        dt = np.dtype([("c" + str(i), np.uint8) for i in range(c)])
        structured = np.frombuffer(flat.tobytes(), dtype=dt)
        unique_vals, counts = np.unique(structured, return_counts=True)
        if len(unique_vals) < 2:
            return np.asarray([255] * c)
        order = np.argsort(-counts)
        fg = unique_vals[order[1]]
        pixel_id = [int(fg[f"c{i}"]) for i in range(c)]
    return np.asarray(pixel_id)


@dataclass
class ObjectInfo:
    """Information about a detected object/cell."""

    area: int
    centroid: Tuple[float, float]  # (y, x)
    coords: np.ndarray  # Shape (N, 2) in (y, x) order


class SegmentationAdapter(ABC):
    """Abstract base class for segmentation format adapters.

    Each adapter knows how to:
    1. Load a segmentation image from disk
    2. Create a binary mask from its format
    3. Extract individual objects with their properties
    4. Detect the foreground pixel ID (if applicable)
    """

    name: str = "base"

    def load(self, path: str) -> np.ndarray:
        """Load a segmentation image from disk.

        Supports standard image formats (PNG, TIFF, etc.) and .dzip files.
        Subclasses can override this to support additional formats.

        Parameters
        ----------
        path : str
            Path to the segmentation file.

        Returns
        -------
        np.ndarray
            The loaded segmentation image.
        """
        if path.endswith(".dzip"):
            return reconstruct_dzi(path)
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    @abstractmethod
    def create_binary_mask(
        self,
        segmentation: np.ndarray,
        pixel_id: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Create a binary mask from the segmentation.

        Args:
            segmentation: The loaded segmentation image.
            pixel_id: Optional pixel color to match (for color-based formats).

        Returns:
            Boolean 2D array where True indicates foreground.
        """
        pass

    @abstractmethod
    def extract_objects(
        self,
        segmentation: np.ndarray,
        binary_mask: np.ndarray,
        min_area: int = 0,
    ) -> List[ObjectInfo]:
        """Extract individual objects from the segmentation.

        Args:
            segmentation: The loaded segmentation image.
            binary_mask: The binary mask from create_binary_mask().
            min_area: Minimum object area threshold.

        Returns:
            List of ObjectInfo for each detected object.
        """
        pass

    def detect_pixel_id(self, segmentation: np.ndarray) -> Optional[List[int]]:
        """Auto-detect the foreground pixel ID.

        Override this method for formats that use color-coded pixels.
        Default implementation returns None (not applicable).

        Args:
            segmentation: The loaded segmentation image.

        Returns:
            List of [R, G, B] values or None if not applicable.
        """
        return None




class BinaryAdapter(SegmentationAdapter):
    """Adapter for binary/color-based segmentation masks (ilastik format).

    Used for ilastik output and other tools that produce binary/color masks
    where foreground is identified by a specific pixel color.
    """

    name: str = "binary"

    def create_binary_mask(
        self,
        segmentation: np.ndarray,
        pixel_id: Optional[List[int]] = None,
    ) -> np.ndarray:
        if pixel_id is None:
            pixel_id = self.detect_pixel_id(segmentation)

        if segmentation.ndim == 2:
            return segmentation == pixel_id[0]
        else:
            mask = segmentation[:, :, 0] == pixel_id[0]
            if segmentation.shape[2] > 1:
                mask &= segmentation[:, :, 1] == pixel_id[1]
            if segmentation.shape[2] > 2:
                mask &= segmentation[:, :, 2] == pixel_id[2]
            return mask

    def extract_objects(
        self,
        segmentation: np.ndarray,
        binary_mask: np.ndarray,
        min_area: int = 0,
    ) -> List[ObjectInfo]:
        from ..pipeline.connected_components import connected_components_props

        props = connected_components_props(binary_mask, connectivity=4)
        return [
            ObjectInfo(area=p["area"], centroid=p["centroid"], coords=p["coords"])
            for p in props
            if p["area"] > min_area
        ]

    def detect_pixel_id(self, segmentation: np.ndarray) -> Optional[List[int]]:
        """Detect the most common non-black pixel color."""
        return _detect_pixel_id(segmentation)


class CellposeAdapter(SegmentationAdapter):
    """Adapter for Cellpose instance segmentation outputs.

    Cellpose produces images where each unique non-zero value
    represents a different cell instance.
    """

    name: str = "cellpose"

    def create_binary_mask(
        self,
        segmentation: np.ndarray,
        pixel_id: Optional[List[int]] = None,
    ) -> np.ndarray:
        # pixel_id is ignored for labeled images
        if segmentation.ndim == 2:
            return segmentation != 0
        else:
            return segmentation[:, :, 0] != 0

    def extract_objects(
        self,
        segmentation: np.ndarray,
        binary_mask: np.ndarray,
        min_area: int = 0,
    ) -> List[ObjectInfo]:
        from ..pipeline.connected_components import labeled_image_props

        props = labeled_image_props(segmentation)
        return [
            ObjectInfo(area=p["area"], centroid=p["centroid"], coords=p["coords"])
            for p in props
            if p["area"] > min_area
        ]


class SegmentationAdapterRegistry:
    """Registry for segmentation format adapters.

    Use this to register new adapters and retrieve them by name.

    Example:
        >>> adapter = SegmentationAdapterRegistry.get("cellpose")
        >>> binary_mask = adapter.create_binary_mask(segmentation)
    """

    _adapters: Dict[str, Type[SegmentationAdapter]] = {}

    @classmethod
    def register(cls, adapter_class: Type[SegmentationAdapter]) -> None:
        """Register a new segmentation adapter."""
        cls._adapters[adapter_class.name] = adapter_class

    @classmethod
    def get(cls, name: str) -> SegmentationAdapter:
        """Get an adapter instance by name."""
        if name not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(
                f"Unknown segmentation format '{name}'. Available: {available}"
            )
        return cls._adapters[name]()


# Register built-in adapters
SegmentationAdapterRegistry.register(BinaryAdapter)
SegmentationAdapterRegistry.register(CellposeAdapter)

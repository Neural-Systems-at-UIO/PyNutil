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

import numpy as np


@dataclass
class ObjectInfo:
    """Information about a detected object/cell."""

    area: int
    centroid: Tuple[float, float]  # (y, x)
    coords: np.ndarray  # Shape (N, 2) in (y, x) order


class SegmentationAdapter(ABC):
    """Abstract base class for segmentation format adapters.

    Each adapter knows how to:
    1. Create a binary mask from its format
    2. Extract individual objects with their properties
    3. Detect the foreground pixel ID (if applicable)
    """

    name: str = "base"

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

    def needs_pixel_id(self) -> bool:
        """Whether this format requires a pixel_id to identify foreground.

        Returns:
            True if pixel_id is needed, False otherwise.
        """
        return False


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
        from ..pipeline.image_loaders import detect_pixel_id
        return detect_pixel_id(segmentation)

    def needs_pixel_id(self) -> bool:
        return True


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

    def needs_pixel_id(self) -> bool:
        return False


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
                f"Unknown segmentation format '{name}'. "
                f"Available: {available}"
            )
        return cls._adapters[name]()

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered adapter names."""
        return list(cls._adapters.keys())


# Register built-in adapters
SegmentationAdapterRegistry.register(BinaryAdapter)
SegmentationAdapterRegistry.register(CellposeAdapter)

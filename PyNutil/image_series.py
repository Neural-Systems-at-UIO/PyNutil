"""Image series and section data containers.

These classes represent a series of sections (images or segmentations) to be
processed through the PyNutil pipeline.  Users with custom segmentation types
can construct :class:`Section` and :class:`ImageSeries` objects directly,
providing their own ``numpy`` arrays instead of reading from disk.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Section:
    """A single section, identified by number and backed by an image.

    Either *image* or *path* must be provided.  When only *path* is given the
    image is loaded lazily the first time it is needed by the pipeline.

    Parameters
    ----------
    section_number:
        Numeric identifier that must match a section in the alignment JSON.
    filename:
        Display name used in :attr:`~PyNutil.ExtractionResult.section_filenames`.
        Defaults to an empty string; set explicitly or use the factory functions
        :func:`~PyNutil.read_segmentation_dir` / :func:`~PyNutil.read_image_dir`
        which populate this from the file path automatically.
    image:
        Pre-loaded image array (2-D or 3-D ``numpy`` array).  Provide this
        when you have already loaded or generated the image data yourself.
    path:
        Path to the image file on disk.  The image is loaded on demand by the
        configured segmentation adapter when the section is processed.
    """

    section_number: int
    filename: str = ""
    image: Optional[np.ndarray] = field(default=None, repr=False)
    path: Optional[str] = None

    def __post_init__(self):
        if self.image is None and self.path is None:
            raise ValueError(
                f"Section {self.section_number}: either 'image' or 'path' must be provided."
            )

    def get_image(self, adapter) -> np.ndarray:
        """Return the image array, loading from *path* if not pre-loaded.

        Parameters
        ----------
        adapter:
            A :class:`~PyNutil.processing.adapters.segmentation.SegmentationAdapter`
            used to load the file when *image* is ``None``.
        """
        if self.image is not None:
            return self.image
        if self.path is not None:
            return adapter.load(self.path)
        raise ValueError(
            f"Section {self.section_number} has neither image data nor a file path."
        )


@dataclass
class ImageSeries:
    """An ordered collection of :class:`Section` objects.

    Construct this directly when you want to supply custom image data, or use
    :func:`~PyNutil.read_segmentation_dir` / :func:`~PyNutil.read_image_dir`
    to build one from a folder of image files.

    Parameters
    ----------
    sections:
        List of :class:`Section` objects.  Order does not matter; sections are
        looked up by ``section_number``.
    pixel_id:
        RGB value (or label) identifying the segmented class of interest.
        Set by :func:`~PyNutil.read_segmentation_dir` and consumed by
        :func:`~PyNutil.seg_to_coords`.
    segmentation_format:
        Name of the segmentation adapter to use (e.g. ``"binary"`` or
        ``"cellpose"``).  Set by :func:`~PyNutil.read_segmentation_dir` and
        consumed by :func:`~PyNutil.seg_to_coords`.
    """

    sections: List[Section] = field(default_factory=list)
    pixel_id: object = field(default_factory=lambda: [0, 0, 0])
    segmentation_format: str = "binary"
    _section_map: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        seen = {}
        for s in self.sections:
            if s.section_number in seen:
                warnings.warn(
                    f"Duplicate section_number {s.section_number}: "
                    f"'{seen[s.section_number]}' and '{s.filename}'. "
                    f"Only '{seen[s.section_number]}' will be used."
                )
            seen[s.section_number] = s.filename
        self._section_map = {s.section_number: s for s in self.sections}

    def get_section_nr(self, section_number: int) -> Optional[Section]:
        """Return the :class:`Section` whose ``section_number`` matches, or ``None``."""
        return self._section_map.get(section_number)

    @property
    def filenames(self) -> List[str]:
        """Display filenames for all sections."""
        return [s.filename for s in self.sections]

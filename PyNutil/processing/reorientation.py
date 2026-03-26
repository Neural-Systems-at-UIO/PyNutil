"""Utilities for reorienting coordinates and volumes to a target orientation."""

import numpy as np
from brainglobe_space import AnatomicalSpace

# The internal orientation used by PyNutil after process_atlas_volume
# (transpose([2,0,1])[::-1,::-1,::-1] applied to BrainGlobe's native "asr").
INTERNAL_ORIENTATION = "lpi"


def reorient_points(points, atlas_shape, target_orientation):
    """Reorient (N, 3) atlas-space points from internal to target orientation.

    Args:
        points: (N, 3) array of coordinates in internal orientation.
        atlas_shape: Shape of the atlas volume in internal orientation.
        target_orientation: 3-letter BrainGlobe orientation string (e.g. "asr").

    Returns:
        (N, 3) array of coordinates in the target orientation.
    """
    if points is None or len(points) == 0:
        return points
    source = AnatomicalSpace(INTERNAL_ORIENTATION, shape=atlas_shape)
    target = AnatomicalSpace(target_orientation)
    return source.map_points_to(target, points)


def reorient_volume(volume, atlas_shape, target_orientation):
    """Reorient a 3D volume from internal to target orientation.

    Args:
        volume: 3D numpy array in internal orientation.
        atlas_shape: Shape of the atlas volume in internal orientation.
        target_orientation: 3-letter BrainGlobe orientation string (e.g. "asr").

    Returns:
        3D numpy array in the target orientation.
    """
    source = AnatomicalSpace(INTERNAL_ORIENTATION, shape=atlas_shape)
    target = AnatomicalSpace(target_orientation)
    return source.map_stack_to(target, volume)

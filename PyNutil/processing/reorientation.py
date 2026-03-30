"""Utilities for reorienting coordinates and volumes to a target orientation."""

import numpy as np
from brainglobe_space import AnatomicalSpace

# The internal orientation used by PyNutil after process_atlas_volume
# (transpose([2,0,1])[::-1,::-1,::-1] applied to BrainGlobe's native "asr").
INTERNAL_ORIENTATION = "lpi"
_ORIENTATION_GROUPS = ("lr", "si", "ap")


def _validate_orientation_code(orientation):
    """Validate a 3-letter BrainGlobe orientation code.

    A valid code must contain exactly one axis from each anatomical pair:
    left/right, superior/inferior, and anterior/posterior.
    Examples: ``"lpi"``, ``"ras"``, ``"asr"``.
    """
    if not isinstance(orientation, str):
        raise ValueError(
            "Invalid orientation code: expected a 3-letter string such as "
            "'lpi', 'ras', or 'asr'."
        )

    normalized = orientation.lower()
    if len(normalized) != 3:
        raise ValueError(
            f"Invalid orientation code {orientation!r}: expected exactly 3 letters. "
            "Use one first letter from each axis pair: left/right, superior/inferior, "
            "and anterior/posterior, for example 'lpi', 'ras', or 'asr'."
        )

    invalid_letters = sorted({ch for ch in normalized if ch not in "lrsiap"})
    if invalid_letters:
        raise ValueError(
            f"Invalid orientation code {orientation!r}: unsupported letter(s) "
            f"{invalid_letters}. Use one letter from each axis pair: left/right, "
            "superior/inferior, and anterior/posterior."
        )

    missing_groups = []
    repeated_groups = []
    for group in _ORIENTATION_GROUPS:
        count = sum(ch in group for ch in normalized)
        if count == 0:
            missing_groups.append(group)
        elif count > 1:
            repeated_groups.append(group)

    if missing_groups or repeated_groups:
        pieces = []
        if repeated_groups:
            pieces.append(
                "repeats axis pair(s) "
                + ", ".join(f"{group[0]}/{group[1]}" for group in repeated_groups)
            )
        if missing_groups:
            pieces.append(
                "is missing axis pair(s) "
                + ", ".join(f"{group[0]}/{group[1]}" for group in missing_groups)
            )
        raise ValueError(
            f"Invalid orientation code {orientation!r}: " + "; ".join(pieces) + ". "
            "A valid code uses exactly one letter from each axis pair: left/right, "
            "superior/inferior, and anterior/posterior, for example 'lpi', 'ras', "
            "or 'asr'."
        )

    return normalized


def _shape_in_orientation(internal_shape, orientation):
    """Derive atlas shape in a given orientation from the internal shape.

    Args:
        internal_shape: Shape of the atlas volume in INTERNAL_ORIENTATION.
        orientation: Target orientation string.

    Returns:
        Tuple of ints — the shape the atlas would have in *orientation*.
    """
    orientation = _validate_orientation_code(orientation)
    if orientation == INTERNAL_ORIENTATION:
        return internal_shape
    source = AnatomicalSpace(INTERNAL_ORIENTATION, shape=internal_shape)
    target = AnatomicalSpace(orientation)
    mat = source.transformation_matrix_to(target)
    perm = np.abs(mat[:3, :3]).argmax(axis=1)
    return tuple(internal_shape[p] for p in perm)


def reorient_points(points, internal_atlas_shape, target_orientation,
                    source_orientation=None):
    """Reorient (N, 3) atlas-space points between orientations.

    Args:
        points: (N, 3) array of coordinates.
        internal_atlas_shape: Shape of the atlas volume in INTERNAL_ORIENTATION.
        target_orientation: 3-letter BrainGlobe orientation string (e.g. "asr").
        source_orientation: Source orientation. Defaults to INTERNAL_ORIENTATION.

    Returns:
        (N, 3) array of coordinates in the target orientation.
    """
    if points is None or len(points) == 0:
        return points
    target_orientation = _validate_orientation_code(target_orientation)
    if source_orientation is None:
        source_orientation = INTERNAL_ORIENTATION
    source_orientation = _validate_orientation_code(source_orientation)
    source_shape = _shape_in_orientation(internal_atlas_shape, source_orientation)
    source = AnatomicalSpace(source_orientation, shape=source_shape)
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
    target_orientation = _validate_orientation_code(target_orientation)
    source = AnatomicalSpace(INTERNAL_ORIENTATION, shape=atlas_shape)
    target = AnatomicalSpace(target_orientation)
    return source.map_stack_to(target, volume)

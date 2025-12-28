"""Section visualisation utilities.

Generates colored atlas slice PNGs and optionally overlays segmentation pixels.
"""

import os
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd

from ..processing.generate_target_slice import generate_target_slice
from ..io.read_and_write import load_segmentation


def _build_color_lookup(
    atlas_labels: pd.DataFrame,
    *,
    default_colour: Tuple[int, int, int] = (128, 128, 128),
    max_direct_lut_id: int = 2_000_000,
) -> Tuple[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], Tuple[int, int, int]]:
    """Build a fast color lookup for atlas IDs.

    Returns:
        (mode, lookup, default_colour)

    mode:
        - "direct": lookup is a (max_id+1, 3) uint8 LUT; index directly by atlas ID
        - "sorted": lookup is (ids_sorted, rgb_sorted); use searchsorted

    This keeps behaviour consistent with the previous implementation:
    - atlas id 0 maps to black
    - unmapped ids map to grey (default_colour)
    """
    if atlas_labels is None or len(atlas_labels) == 0:
        lut = np.empty((1, 3), dtype=np.uint8)
        lut[0] = (0, 0, 0)
        return "direct", lut, default_colour

    ids = atlas_labels["idx"].to_numpy(dtype=np.int64, copy=False)
    rgb = atlas_labels[["r", "g", "b"]].to_numpy(dtype=np.uint8, copy=False)
    max_id = int(ids.max(initial=0)) if ids.size else 0

    if 0 <= max_id <= max_direct_lut_id:
        lut = np.empty((max_id + 1, 3), dtype=np.uint8)
        lut[:] = default_colour
        lut[0] = (0, 0, 0)
        # If duplicate ids exist, later rows overwrite earlier ones (same as dict assignment)
        lut[ids] = rgb
        return "direct", lut, default_colour

    order = np.argsort(ids)
    ids_sorted = ids[order]
    rgb_sorted = rgb[order]
    return "sorted", (ids_sorted, rgb_sorted), default_colour


def create_colored_image_from_slice(
    atlas_slice: np.ndarray,
    lookup_mode: str,
    lookup: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    default_colour: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Create an RGB image from an atlas slice using a fast lookup."""
    atlas_ids = atlas_slice.astype(np.int64, copy=False)
    height, width = atlas_ids.shape
    out = np.empty((height, width, 3), dtype=np.uint8)
    out[:] = default_colour

    if lookup_mode == "direct":
        lut = lookup  # type: ignore[assignment]
        max_id = lut.shape[0] - 1
        valid = (atlas_ids >= 0) & (atlas_ids <= max_id)
        if np.any(valid):
            out[valid] = lut[atlas_ids[valid]]
        return out

    ids_sorted, rgb_sorted = lookup  # type: ignore[misc]
    idx = np.searchsorted(ids_sorted, atlas_ids)
    valid = (idx < ids_sorted.size) & (ids_sorted[idx] == atlas_ids)
    if np.any(valid):
        out[valid] = rgb_sorted[idx[valid]]
    return out


def overlay_segmentation_on_rgb(
    rgb_image: np.ndarray,
    segmentation_path: str,
    *,
    segmentation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlay segmentation on an RGB NumPy image.

    This inverts the underlying atlas RGB values at segmentation pixels for
    maximum contrast.
    """
    try:
        if segmentation is None:
            segmentation = load_segmentation(segmentation_path)

        height, width = rgb_image.shape[:2]
        segmentation = cv2.resize(
            segmentation, (width, height), interpolation=cv2.INTER_NEAREST
        )

        if segmentation.ndim == 3:
            seg_grey = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
        else:
            seg_grey = segmentation

        seg_u8 = seg_grey.astype(np.uint8, copy=False)
        hist = np.bincount(seg_u8.reshape(-1), minlength=256)
        background_val = int(hist.argmax())
        mask = seg_u8 != background_val
        if not np.any(mask):
            return rgb_image

        out = rgb_image.copy()
        out[mask] = 255 - out[mask]
        return out
    except Exception as e:
        print(f"Warning: Could not overlay segmentation from {segmentation_path}: {e}")
        return rgb_image


def create_colored_atlas_slice(
    slice_dict: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_path: str,
    segmentation_path: Optional[str] = None,
    objects_data: Optional[List[Dict]] = None,
    scale_factor: float = 0.5,
    _color_lookup: Optional[
        Tuple[
            str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], Tuple[int, int, int]
        ]
    ] = None,
) -> None:
    """Create a coloured atlas slice and optionally overlay segmentation pixels."""
    atlas_slice = generate_target_slice(slice_dict["anchoring"], atlas_volume)
    if _color_lookup is None:
        lookup_mode, lookup, default_colour = _build_color_lookup(atlas_labels)
    else:
        lookup_mode, lookup, default_colour = _color_lookup
    coloured_slice = create_colored_image_from_slice(
        atlas_slice,
        lookup_mode,
        lookup,
        default_colour,
    )

    target_width = coloured_slice.shape[1]
    target_height = coloured_slice.shape[0]

    segmentation_img = None
    if segmentation_path and os.path.exists(segmentation_path):
        try:
            segmentation_img = load_segmentation(segmentation_path)
            seg_height, seg_width = segmentation_img.shape[:2]
            target_width, target_height = seg_width, seg_height
        except Exception as e:
            print(f"Warning: Could not load segmentation for sizing: {e}")

    if segmentation_img is None:
        try:
            reg_w = int(slice_dict.get("width", target_width))
            reg_h = int(slice_dict.get("height", target_height))
            if reg_w > 0 and reg_h > 0:
                target_width, target_height = reg_w, reg_h
        except Exception:
            pass

    if (coloured_slice.shape[1], coloured_slice.shape[0]) != (
        target_width,
        target_height,
    ):
        coloured_slice = cv2.resize(
            coloured_slice,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    if scale_factor != 1.0:
        new_width = max(1, int(target_width * scale_factor))
        new_height = max(1, int(target_height * scale_factor))
        coloured_slice = cv2.resize(
            coloured_slice, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

    if segmentation_path and os.path.exists(segmentation_path):
        coloured_slice = overlay_segmentation_on_rgb(
            coloured_slice, segmentation_path, segmentation=segmentation_img
        )

    # Object overlay intentionally not implemented yet in PyNutil core (requires passing object coords)
    # Kept as a hook for future usage.
    _ = objects_data

    # OpenCV expects BGR ordering when saving.
    bgr = cv2.cvtColor(coloured_slice, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(output_path, bgr):
        raise RuntimeError(f"Failed to write visualisation image: {output_path}")


def create_section_visualisations(
    segmentation_folder: str,
    alignment_json: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_folder: str,
    objects_per_section: Optional[List[List[Dict]]] = None,
    scale_factor: float = 0.5,
):
    """Create visualisation images for all sections in the analysis."""
    viz_dir = os.path.join(output_folder, "visualisations")
    os.makedirs(viz_dir, exist_ok=True)

    # Pre-index segmentations once to avoid repeated exists checks.
    seg_index: Dict[str, str] = {}
    ext_priority = {".png": 0, ".tif": 1, ".tiff": 2, ".jpg": 3, ".jpeg": 4}
    try:
        for filename in os.listdir(segmentation_folder):
            base, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in ext_priority:
                continue
            path = os.path.join(segmentation_folder, filename)
            existing = seg_index.get(base)
            if existing is None:
                seg_index[base] = path
                continue
            _, existing_ext = os.path.splitext(existing)
            if ext_priority.get(ext, 999) < ext_priority.get(existing_ext.lower(), 999):
                seg_index[base] = path
    except Exception:
        # Fall back to per-slice existence checks if indexing fails.
        seg_index = {}

    color_lookup = _build_color_lookup(atlas_labels)

    slices = alignment_json.get("slices", [])
    for i, slice_dict in enumerate(slices):
        try:
            filename = slice_dict.get("filename", "")
            base_name = os.path.splitext(filename)[0] if filename else f"slice_{i:03d}"

            segmentation_path = None
            if filename and seg_index:
                segmentation_path = seg_index.get(base_name)
            elif filename:
                candidates = [
                    f"{base_name}.png",
                    f"{base_name}.tif",
                    f"{base_name}.tiff",
                    f"{base_name}.jpg",
                    f"{base_name}.jpeg",
                ]
                for candidate in candidates:
                    p = os.path.join(segmentation_folder, candidate)
                    if os.path.exists(p):
                        segmentation_path = p
                        break

            section_objects = None
            if objects_per_section and i < len(objects_per_section):
                section_objects = objects_per_section[i]

            output_filename = (
                f"section_{slice_dict.get('nr', i):03d}_{base_name}_atlas_colored.png"
            )
            output_path = os.path.join(viz_dir, output_filename)

            # Ensure the directory for the output file exists (in case base_name has slashes)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            create_colored_atlas_slice(
                slice_dict=slice_dict,
                atlas_volume=atlas_volume,
                atlas_labels=atlas_labels,
                output_path=output_path,
                segmentation_path=segmentation_path,
                objects_data=section_objects,
                scale_factor=scale_factor,
                _color_lookup=color_lookup,
            )

            print(f"Created visualisation: {output_filename}")
        except Exception as e:
            print(f"Error creating visualisation for slice {i}: {e}")

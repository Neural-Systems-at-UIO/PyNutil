"""Section visualization utilities.

Generates colored atlas slice PNGs and optionally overlays segmentation contours.
"""

import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ..processing.generate_target_slice import generate_target_slice
from ..io.read_and_write import load_segmentation


def create_atlas_color_map(atlas_labels: pd.DataFrame) -> Dict[int, Tuple[int, int, int]]:
    """Create a color mapping from atlas labels DataFrame."""
    color_map: Dict[int, Tuple[int, int, int]] = {0: (0, 0, 0)}
    for _, row in atlas_labels.iterrows():
        if "idx" in row and "r" in row and "g" in row and "b" in row:
            region_id = int(row["idx"])
            color_map[region_id] = (int(row["r"]), int(row["g"]), int(row["b"]))
    return color_map


def create_colored_image_from_slice(
    atlas_slice: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """Create an RGB image from an atlas slice using the color mapping."""
    height, width = atlas_slice.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    for region_id, color in color_map.items():
        colored_image[atlas_slice == region_id] = color

    # For unmapped regions, use a default gray color
    unmapped_mask = np.isin(atlas_slice, list(color_map.keys()), invert=True)
    colored_image[unmapped_mask] = (128, 128, 128)
    return colored_image


def overlay_segmentation(
    pil_image: Image.Image,
    segmentation_path: str,
    alpha: float = 1.0,
) -> None:
    """Overlay segmentation pixels on top of an atlas slice image.

    This uses a background-value heuristic (most frequent grayscale intensity) to
    build a foreground mask, then overlays those pixels.

    With the default `alpha=1.0`, this inverts the underlying atlas RGB values
    at segmentation pixels for maximum contrast.
    """
    try:
        segmentation = load_segmentation(segmentation_path)

        img_width, img_height = pil_image.size
        segmentation = cv2.resize(
            segmentation, (img_width, img_height), interpolation=cv2.INTER_NEAREST
        )

        # Convert to grayscale for background detection
        if segmentation.ndim == 3:
            # load_segmentation may return RGB; OpenCV expects BGR but for gray conversion it’s fine
            seg_gray = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
        else:
            seg_gray = segmentation

        # Determine background value (most frequent intensity).
        # This makes overlays work for both black-background and white-background segmentations.
        seg_u8 = seg_gray.astype(np.uint8, copy=False)
        hist = np.bincount(seg_u8.reshape(-1), minlength=256)
        background_val = int(hist.argmax())

        # Foreground = anything different from background
        mask = seg_u8 != background_val
        if not np.any(mask):
            return

        # Solid overlay: invert underlying atlas colors at segmentation pixels.
        base_arr = np.array(pil_image.convert("RGB"), copy=True)
        inverted = 255 - base_arr

        if alpha >= 1.0:
            base_arr[mask] = inverted[mask]
        else:
            # Optional blended overlay (kept for flexibility)
            a = float(alpha)
            base_arr[mask] = (
                base_arr[mask].astype(np.float32) * (1.0 - a)
                + inverted[mask].astype(np.float32) * a
            ).round().clip(0, 255).astype(np.uint8)

        pil_image.paste(Image.fromarray(base_arr, mode="RGB"))
    except Exception as e:
        print(f"Warning: Could not overlay segmentation from {segmentation_path}: {e}")


def create_colored_atlas_slice(
    slice_dict: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_path: str,
    segmentation_path: Optional[str] = None,
    objects_data: Optional[List[Dict]] = None,
    scale_factor: float = 0.5,
) -> None:
    """Create a colored atlas slice and optionally overlay segmentation contours."""
    atlas_slice = generate_target_slice(slice_dict["anchoring"], atlas_volume)
    color_map = create_atlas_color_map(atlas_labels)
    colored_slice = create_colored_image_from_slice(atlas_slice, color_map)

    target_width = colored_slice.shape[1]
    target_height = colored_slice.shape[0]

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

    if (colored_slice.shape[1], colored_slice.shape[0]) != (target_width, target_height):
        colored_slice = cv2.resize(
            colored_slice, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )

    if scale_factor != 1.0:
        new_width = max(1, int(target_width * scale_factor))
        new_height = max(1, int(target_height * scale_factor))
        colored_slice = cv2.resize(
            colored_slice, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

    pil_image = Image.fromarray(colored_slice)

    if segmentation_path and os.path.exists(segmentation_path):
        overlay_segmentation(pil_image, segmentation_path)

    # Object overlay intentionally not implemented yet in PyNutil core (requires passing object coords)
    # Kept as a hook for future usage.
    _ = objects_data

    pil_image.save(output_path)


def create_section_visualizations(
    segmentation_folder: str,
    alignment_json: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_folder: str,
    objects_per_section: Optional[List[List[Dict]]] = None,
    scale_factor: float = 0.5,
):
    """Create visualization images for all sections in the analysis."""
    viz_dir = os.path.join(output_folder, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    slices = alignment_json.get("slices", [])
    for i, slice_dict in enumerate(slices):
        try:
            filename = slice_dict.get("filename", "")
            base_name = os.path.splitext(filename)[0] if filename else f"slice_{i:03d}"

            segmentation_path = None
            if filename:
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

            create_colored_atlas_slice(
                slice_dict=slice_dict,
                atlas_volume=atlas_volume,
                atlas_labels=atlas_labels,
                output_path=output_path,
                segmentation_path=segmentation_path,
                objects_data=section_objects,
                scale_factor=scale_factor,
            )

            print(f"Created visualization: {output_filename}")
        except Exception as e:
            print(f"Error creating visualization for slice {i}: {e}")

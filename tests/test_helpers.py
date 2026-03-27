import json
import os
import shutil

from brainglobe_atlasapi import BrainGlobeAtlas

from PyNutil import (
    load_custom_atlas,
    read_alignment,
    seg_to_coords,
    image_to_coords,
    xy_to_coords,
    quantify_coords,
)
from PyNutil.io.atlas_loader import resolve_atlas


def small_volume_scale(atlas_shape, target_max_dim: float = 80.0) -> float:
    sx, sy, sz = (int(x) for x in atlas_shape)
    max_dim = max(sx, sy, sz)
    return min(float(target_max_dim) / float(max_dim), 1.0)


def load_atlas_from_settings(settings: dict):
    """Load an AtlasData from a settings dictionary."""
    if settings.get("atlas_name"):
        return resolve_atlas(BrainGlobeAtlas(settings["atlas_name"]))
    return load_custom_atlas(
        settings["atlas_path"],
        settings.get("hemi_path"),
        settings["label_path"],
    )


def run_pipeline_from_settings(settings: dict):
    """Run the full extraction + quantification pipeline from a settings dict.

    Returns (atlas, result, label_df, alignment).
    """
    atlas = load_atlas_from_settings(settings)
    alignment = read_alignment(settings["alignment_json"])

    if settings.get("coordinate_file"):
        import pandas as pd
        result = xy_to_coords(
            pd.read_csv(settings["coordinate_file"]),
            alignment,
            atlas,
        )
    elif settings.get("image_folder"):
        result = image_to_coords(
            settings["image_folder"],
            alignment,
            atlas,
            intensity_channel=settings.get("intensity_channel", "grayscale"),
            min_intensity=settings.get("min_intensity"),
            max_intensity=settings.get("max_intensity"),
        )
    else:
        result = seg_to_coords(
            settings["segmentation_folder"],
            alignment,
            atlas,
            pixel_id=settings.get("colour", [0, 0, 0]),
            segmentation_format=settings.get("segmentation_format", "binary"),
        )

    label_df = quantify_coords(result, atlas)
    return atlas, result, label_df, alignment


def run_pipeline_from_settings_file(settings_path: str):
    """Load settings from JSON and run the pipeline."""
    with open(settings_path) as f:
        settings = json.load(f)
    return run_pipeline_from_settings(settings)


def copy_tree_to_demo(output_dir: str, demo_dir: str) -> None:
    os.makedirs(os.path.dirname(demo_dir), exist_ok=True)
    shutil.rmtree(demo_dir, ignore_errors=True)
    shutil.copytree(output_dir, demo_dir)

import json
import os
import shutil

from PyNutil import PyNutil


def small_volume_scale(atlas_shape, target_max_dim: float = 80.0) -> float:
    sx, sy, sz = (int(x) for x in atlas_shape)
    max_dim = max(sx, sy, sz)
    return min(float(target_max_dim) / float(max_dim), 1.0)


def pynutil_from_settings_dict(settings: dict) -> PyNutil:
    """Create a PyNutil instance with atlas settings from a settings dictionary.

    Data-pipeline arguments (segmentation folder, alignment JSON, colour, etc.)
    must be passed to :meth:`~PyNutil.get_coordinates` separately.  Use
    :func:`get_coordinates_kwargs` to extract them from the same dict.
    """
    return PyNutil(
        atlas_name=settings.get("atlas_name"),
        atlas_path=settings.get("atlas_path"),
        label_path=settings.get("label_path"),
        hemi_path=settings.get("hemi_path"),
        voxel_size_um=settings.get("voxel_size_um"),
    )


def get_coordinates_kwargs(settings: dict) -> dict:
    """Extract data-pipeline kwargs for :meth:`~PyNutil.get_coordinates` from a settings dict."""
    return {
        "segmentation_folder": settings.get("segmentation_folder"),
        "image_folder": settings.get("image_folder"),
        "coordinate_file": settings.get("coordinate_file"),
        "alignment_json": settings.get("alignment_json"),
        "colour": settings.get("colour"),
        "intensity_channel": settings.get("intensity_channel"),
        "min_intensity": settings.get("min_intensity"),
        "max_intensity": settings.get("max_intensity"),
        "segmentation_format": settings.get("segmentation_format", "binary"),
        "custom_region_path": settings.get("custom_region_path"),
    }


def make_pynutil_ready(settings_path: str) -> PyNutil:
    with open(settings_path) as f:
        settings = json.load(f)
    pnt = pynutil_from_settings_dict(settings)
    pnt.get_coordinates(**get_coordinates_kwargs(settings), object_cutoff=0)
    pnt.quantify_coordinates()
    return pnt


def copy_tree_to_demo(output_dir: str, demo_dir: str) -> None:
    os.makedirs(os.path.dirname(demo_dir), exist_ok=True)
    shutil.rmtree(demo_dir, ignore_errors=True)
    shutil.copytree(output_dir, demo_dir)

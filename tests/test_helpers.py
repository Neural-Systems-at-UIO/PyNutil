import os
import shutil

from PyNutil import PyNutil


def small_volume_scale(atlas_shape, target_max_dim: float = 80.0) -> float:
    sx, sy, sz = (int(x) for x in atlas_shape)
    max_dim = max(sx, sy, sz)
    return min(float(target_max_dim) / float(max_dim), 1.0)


def make_pynutil_ready(settings_path: str) -> PyNutil:
    pnt = PyNutil(settings_file=settings_path)
    pnt.get_coordinates(object_cutoff=0)
    pnt.quantify_coordinates()
    return pnt


def copy_tree_to_demo(output_dir: str, demo_dir: str) -> None:
    os.makedirs(os.path.dirname(demo_dir), exist_ok=True)
    shutil.rmtree(demo_dir, ignore_errors=True)
    shutil.copytree(output_dir, demo_dir)

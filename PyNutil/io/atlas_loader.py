import brainglobe_atlasapi
import pandas as pd
import numpy as np
import nrrd
from functools import lru_cache

from ..results import AtlasData


def load_atlas_labels(atlas=None, atlas_name=None):
    if atlas_name:
        atlas = brainglobe_atlasapi.BrainGlobeAtlas(atlas_name=atlas_name)
    if not atlas_name and not atlas:
        raise Exception("Either atlas or atlas name must be specified")
    atlas_structures = {
        "idx": [],
        "name": [],
        "r": [],
        "g": [],
        "b": [],
    }
    for structure in atlas.structures_list:
        atlas_structures["idx"].append(structure["id"])
        atlas_structures["name"].append(structure["name"])
        rgb = structure["rgb_triplet"]
        atlas_structures["r"].append(rgb[0])
        atlas_structures["g"].append(rgb[1])
        atlas_structures["b"].append(rgb[2])
    atlas_structures["idx"].insert(0, 0)
    atlas_structures["name"].insert(0, "Clear Label")
    atlas_structures["r"].insert(0, 0)
    atlas_structures["g"].insert(0, 0)
    atlas_structures["b"].insert(0, 0)
    atlas_labels = pd.DataFrame(atlas_structures)
    return atlas_labels


def resolve_atlas(atlas):
    """Convert an atlas argument to AtlasData.

    Accepts an ``AtlasData`` instance (returned as-is) or a
    ``BrainGlobeAtlas``-like object (converted via volume processing and
    label loading).
    """
    if isinstance(atlas, AtlasData):
        return atlas
    # Assume BrainGlobeAtlas-like object
    volume = process_atlas_volume(atlas.annotation)
    hemi_map = process_atlas_volume(atlas.hemispheres)
    labels = load_atlas_labels(atlas)
    voxel_size_um = float(atlas.resolution[0]) if hasattr(atlas, "resolution") else None
    return AtlasData(volume=volume, hemi_map=hemi_map, labels=labels, voxel_size_um=voxel_size_um)


def resolve_atlas_labels(atlas_labels):
    """Resolve atlas labels input into a DataFrame.

    Accepts a raw labels DataFrame, AtlasData-like objects exposing ``labels``,
    or BrainGlobeAtlas-like objects exposing ``structures_list``.
    """
    if isinstance(atlas_labels, pd.DataFrame):
        return atlas_labels
    if hasattr(atlas_labels, "labels"):
        return atlas_labels.labels
    if hasattr(atlas_labels, "structures_list"):
        return load_atlas_labels(atlas_labels)
    raise TypeError(
        "atlas_labels must be a pandas DataFrame, AtlasData-like (.labels), "
        "or BrainGlobeAtlas-like (.structures_list)."
    )


def process_atlas_volume(vol):
    """
    Processes the atlas volume by transposing and reversing axes.

    Parameters
    ----------
    vol : numpy.ndarray
        The atlas volume to process.

    Returns
    -------
    numpy.ndarray
        The processed atlas volume.
    """
    return np.transpose(vol, [2, 0, 1])[::-1, ::-1, ::-1]


@lru_cache(maxsize=8)
def load_custom_atlas(atlas_path, hemi_path, label_path):
    """
    Loads a custom atlas from provided file paths.

    Returns
    -------
    AtlasData
        Bundle containing atlas volume, hemisphere map, and labels.
    """
    atlas_volume, _ = nrrd.read(atlas_path)

    if hemi_path:
        hemi_volume, _ = nrrd.read(hemi_path)
    else:
        hemi_volume = None

    atlas_labels = pd.read_csv(label_path)

    return AtlasData(volume=atlas_volume, hemi_map=hemi_volume, labels=atlas_labels)

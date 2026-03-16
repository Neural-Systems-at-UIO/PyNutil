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
        "idx": [i["id"] for i in atlas.structures_list],
        "name": [i["name"] for i in atlas.structures_list],
        "r": [i["rgb_triplet"][0] for i in atlas.structures_list],
        "g": [i["rgb_triplet"][1] for i in atlas.structures_list],
        "b": [i["rgb_triplet"][2] for i in atlas.structures_list],
    }
    atlas_structures["idx"].insert(0, 0)
    atlas_structures["name"].insert(0, "Clear Label")
    atlas_structures["r"].insert(0, 0)
    atlas_structures["g"].insert(0, 0)
    atlas_structures["b"].insert(0, 0)
    atlas_labels = pd.DataFrame(atlas_structures)
    return atlas_labels


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


@lru_cache(maxsize=8)
def load_atlas_data(atlas_name):
    """
    Loads atlas data using the brainglobe_atlasapi.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to load.

    Returns
    -------
    AtlasData
        Bundle containing atlas volume, hemisphere map, and labels.
    """
    atlas = brainglobe_atlasapi.BrainGlobeAtlas(atlas_name=atlas_name)
    atlas_labels = load_atlas_labels(atlas)
    atlas_volume = process_atlas_volume(atlas.annotation)
    hemi_map = process_atlas_volume(atlas.hemispheres)
    print("atlas labels loaded ✅")
    return AtlasData(volume=atlas_volume, hemi_map=hemi_map, labels=atlas_labels)


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

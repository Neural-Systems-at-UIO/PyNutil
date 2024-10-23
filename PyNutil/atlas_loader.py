import brainglobe_atlasapi
import pandas as pd
import numpy as np
from .read_and_write import read_atlas_volume


def load_atlas_data(atlas_name):
    atlas = brainglobe_atlasapi.BrainGlobeAtlas(atlas_name=atlas_name)
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
    atlas_volume = process_atlas_volume(atlas)
    print("atlas labels loaded ✅")
    return atlas_volume, atlas_labels


def process_atlas_volume(atlas):
    print("reorienting brainglobe atlas into quicknii space...")
    return np.transpose(atlas.annotation, [2, 0, 1])[:, ::-1, ::-1]


def load_custom_atlas(atlas_path, label_path):
    atlas_volume = read_atlas_volume(atlas_path)
    atlas_labels = pd.read_csv(label_path)
    return atlas_volume, atlas_labels

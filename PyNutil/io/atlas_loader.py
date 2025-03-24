import brainglobe_atlasapi
import pandas as pd
import numpy as np
import nrrd


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
    atlas_volume = process_atlas_volume(atlas.annotation)
    hemi_map = process_atlas_volume(atlas.hemispheres)
    print("atlas labels loaded ✅")
    return atlas_volume,hemi_map, atlas_labels


def process_atlas_volume(vol):
    return np.transpose(vol, [2, 0, 1])[::-1, ::-1, ::-1]


def load_custom_atlas(atlas_path, hemi_path, label_path):
    atlas_volume, _ = nrrd.read(atlas_path)
    if hemi_path:
        hemi_volume, _ = nrrd.read(hemi_path)
    else:
        hemi_volume = None
    atlas_labels = pd.read_csv(label_path)
    return atlas_volume, hemi_volume, atlas_labels

import brainglobe_atlasapi
import pandas as pd
import numpy as np
import nrrd

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

def load_atlas_data(atlas_name):
    """
    Loads atlas data using the brainglobe_atlasapi.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to load.

    Returns
    -------
    numpy.ndarray
        The atlas volume array.
    numpy.ndarray
        The hemisphere data array.
    pandas.DataFrame
        A dataframe containing atlas labels and RGB information.
    """
    atlas = brainglobe_atlasapi.BrainGlobeAtlas(atlas_name=atlas_name)
    atlas_labels = load_atlas_labels(atlas)
    atlas_volume = process_atlas_volume(atlas.annotation)
    hemi_map = process_atlas_volume(atlas.hemispheres)
    print("atlas labels loaded ✅")
    return atlas_volume, hemi_map, atlas_labels


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


def load_custom_atlas(atlas_path, hemi_path, label_path):
    """
    Loads a custom atlas from provided file paths.
    Includes optimization for large/non-contiguous indices.
    """
    atlas_volume, _ = nrrd.read(atlas_path)

    if hemi_path:
        hemi_volume, _ = nrrd.read(hemi_path)
    else:
        hemi_volume = None

    atlas_labels = pd.read_csv(label_path)

    # OPTIMIZATION: Remap huge indices to compact range to prevent massive memory allocation
    try:
        original_indices = atlas_labels["idx"].values
        max_idx = original_indices.max()
        unique_indices = np.unique(original_indices)

        # If max index is large (arbitrary threshold 100k), we assume it's sparse/non-contiguous
        if max_idx > 100000:
            print(f"Large atlas indices detected (max={max_idx}). Remapping to compact range...")

            # Get indices actually used in the volume
            volume_indices = np.unique(atlas_volume)

            # Union of labels and volume indices
            all_indices = np.union1d(original_indices, volume_indices)
            max_all = all_indices.max()

            # Create efficient lookup using numpy indexing
            # Default to identity mapping, but size it to the max existing ID
            lookup = np.arange(max_all + 1, dtype=np.uint32)

            # Create compact mapping {Real_ID: Compact_ID}
            compact_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(all_indices)}

            # Update the lookup table
            for old_idx, new_idx in compact_mapping.items():
                lookup[old_idx] = new_idx

            # Remap volume efficiently
            atlas_volume = lookup[atlas_volume]

            # Store original indices in dataframe for reporting later
            atlas_labels["original_idx"] = atlas_labels["idx"]
            atlas_labels["idx"] = atlas_labels["idx"].map(compact_mapping)

    except Exception as e:
        print(f"Warning during index remapping: {e}")

    return atlas_volume, hemi_volume, atlas_labels

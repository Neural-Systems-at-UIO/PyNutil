from .coordinate_extraction import folder_to_atlas_space
from .counting_and_load import label_points


def extract_coordinates(segmentation_folder, alignment_json, atlas_labels, pixel_id, non_linear, object_cutoff, atlas_volume, use_flat):
    return folder_to_atlas_space(
        segmentation_folder,
        alignment_json,
        atlas_labels,
        pixel_id,
        non_linear,
        object_cutoff,
        atlas_volume,
        use_flat,
    )


def label_points_group(points, atlas_volume):
    return label_points(points, atlas_volume, scale_factor=1)

import unittest
import os
import sys

from PyNutil import load_custom_atlas, read_alignment, seg_to_coords

# Add the root directory to sys.path to allow importing PyNutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ATLAS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test_data", "allen_mouse_2017_atlas")
)
ATLAS_PATH = os.path.join(ATLAS_DIR, "annotation_25_reoriented_2017.nrrd")
LABEL_PATH = os.path.join(ATLAS_DIR, "allen2017_colours.csv")


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data", "image_intensity"))
        self.image_folder = os.path.join(self.base_dir, "images")
        self.alignment_json = os.path.join(self.base_dir, "alignment.json")

    def _load_atlas(self):
        return load_custom_atlas(ATLAS_PATH, None, LABEL_PATH)

    def test_intensity_filter_with_segmentation_raises_error(self):
        """seg_to_coords does not accept min_intensity; passing it should raise TypeError."""
        atlas = self._load_atlas()
        alignment = read_alignment(self.alignment_json)
        with self.assertRaises(TypeError):
            seg_to_coords(
                self.image_folder,
                alignment,
                atlas,
                pixel_id=[0, 0, 0],
                min_intensity=10,
            )

    def test_voxel_size_ignored_with_atlas_name(self):
        # We check if it logs a warning. For now we just check if it resets the value.
        # Note: voxel_size_um is ignored when atlas_name is provided; with a custom atlas
        # it is accepted as-is (no inference).  We test the normalisation warning path
        # indirectly via the config's normalize() logic.
        from PyNutil.config import PyNutilConfig
        cfg = PyNutilConfig(atlas_name="allen_mouse_25um", voxel_size_um=10.0)
        cfg.normalize()
        # voxel_size_um should be cleared because atlas_name takes precedence
        self.assertIsNone(cfg.voxel_size_um)


if __name__ == "__main__":
    unittest.main()

import unittest
import os
import sys
from PyNutil import PyNutil

# Add the root directory to sys.path to allow importing PyNutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data", "image_intensity"))
        self.image_folder = os.path.join(self.base_dir, "images")
        self.alignment_json = os.path.join(self.base_dir, "alignment.json")
        self.atlas_name = "allen_mouse_25um"

    def test_both_folders_raises_error(self):
        with self.assertRaises(ValueError) as cm:
            PyNutil(
                segmentation_folder=self.image_folder,
                image_folder=self.image_folder,
                alignment_json=self.alignment_json,
                atlas_name=self.atlas_name
            )
        self.assertIn("not both", str(cm.exception))

    def test_intensity_filter_with_segmentation_raises_error(self):
        with self.assertRaises(ValueError) as cm:
            PyNutil(
                segmentation_folder=self.image_folder,
                alignment_json=self.alignment_json,
                atlas_name=self.atlas_name,
                min_intensity=10
            )
        self.assertIn("only supported when using image_folder", str(cm.exception))

    def test_voxel_size_ignored_with_atlas_name(self):
        # We check if it logs a warning. For now we just check if it resets the value.
        pnt = PyNutil(
            image_folder=self.image_folder,
            alignment_json=self.alignment_json,
            atlas_name=self.atlas_name,
            voxel_size_um=10.0
        )
        # Should be 25.0 (inferred from allen_mouse_25um) not 10.0
        self.assertEqual(pnt.voxel_size_um, 25.0)

if __name__ == "__main__":
    unittest.main()

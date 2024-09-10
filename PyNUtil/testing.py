import unittest
from core.read_and_write import *
from core.main import PyNutil


class testAtlas(unittest.TestCase):
    pnt = PyNutil(
        segmentation_folder="../tests/test_data/big_caudoputamen_test/",
        alignment_json="../tests/test_data/big_caudoputamen.json",
        colour=[0, 0, 0],
        atlas_name="allen_mouse_25um",
    )

    # need help in filling in data for tests
    def test_load_atlas_data(self):
        self.assertTrue(hasattr(self.pnt, "atlas_volume"))
        self.assertTrue(hasattr(self.pnt, "atlas_labels"))

    def test_read_flats(self):
        self.assertEqual(
            len(
                files_in_directory(
                    "../tests/test_data/big_caudoputamen_test/flat_files"
                )
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()

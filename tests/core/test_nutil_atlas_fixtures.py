import os
import unittest


class TestNutilAtlasFixtures(unittest.TestCase):
    def test_required_atlas_fixtures_exist(self):
        tests_dir = os.path.dirname(os.path.dirname(__file__))
        required_files = [
            os.path.join(
                tests_dir,
                "test_data",
                "allen_mouse_2017_atlas",
                "annotation_25_reoriented_2017.nrrd",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "allen_mouse_2017_atlas",
                "allen2017_colours.csv",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "allen_mouse_2015_atlas",
                "labels.nii.gz",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "allen_mouse_2015_atlas",
                "labels.txt",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "waxholm_rat_v4_atlas",
                "labels.nii.gz",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "waxholm_rat_v4_atlas",
                "labels.txt",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "nutil_validator",
                "Q1",
                "Input",
                "Objects_s0218.png",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "nutil_validator",
                "Q3",
                "Input",
                "Segmentation_groundtruth__s006.png",
            ),
            os.path.join(
                tests_dir,
                "test_data",
                "nutil_validator",
                "Q6",
                "correct",
                "Reports",
                "test_RefAtlasRegions",
                "test_RefAtlasRegions.csv",
            ),
        ]

        missing = [path for path in required_files if not os.path.exists(path)]
        self.assertFalse(
            missing,
            "Missing required atlas fixtures in tests/test_data: " + ", ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()

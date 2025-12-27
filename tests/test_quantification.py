import unittest
import os
import sys
import warnings

sys.path.append(os.path.abspath("/home/harryc/github/PyNutil/"))
import numpy as np
import pandas as pd
from PyNutil import PyNutil
import json

from timing_utils import TimedTestCase


class TestQuantification(TimedTestCase):
    def setUp(self):
        self.test_case_dir = os.path.dirname(__file__)
        # Keep test output readable: ignore noisy dependency deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def load_test_case(self, filename):
        test_case_filename = os.path.join(self.test_case_dir, "test_cases", filename)
        with open(test_case_filename, "r") as file:
            return test_case_filename, json.load(file)

    def run_test_case(
        self,
        test_case_filename,
        *,
        create_visualisations: bool = True,
        save_suffix: str = "",
    ):
        test_case_filename, test_case = self.load_test_case(test_case_filename)
        pnt = PyNutil(settings_file=test_case_filename)
        pnt.get_coordinates(object_cutoff=0)
        pnt.quantify_coordinates()
        expected_output_path = os.path.join(
            self.test_case_dir,
            test_case["expected_output_folder"],
            "whole_series_report",
            "counts.csv",
        )
        expected_output = pd.read_csv(expected_output_path, sep=";")
        columns = [
            "r",
            "g",
            "b",
            "damaged_region_area",
            "left_hemi_damaged_region_area",
            "left_hemi_region_area",
            "left_hemi_undamaged_region_area",
            "region_area",
            "right_hemi_damaged_region_area",
            "right_hemi_region_area",
            "right_hemi_undamaged_region_area",
            "undamaged_region_area",
            "damaged_object_count",
            "damaged_pixel_counts",
            "left_hemi_damaged_object_count",
            "left_hemi_damaged_pixel_count",
            "left_hemi_object_count",
            "left_hemi_pixel_count",
            "left_hemi_undamaged_object_count",
            "left_hemi_undamaged_pixel_count",
            "object_count",
            "pixel_count",
            "right_hemi_damaged_object_count",
            "right_hemi_damaged_pixel_count",
            "right_hemi_object_count",
            "right_hemi_pixel_count",
            "right_hemi_undamaged_object_count",
            "right_hemi_undamaged_pixel_count",
            "undamaged_object_count",
            "undamaged_pixel_count",
            "area_fraction",
            "left_hemi_area_fraction",
            "right_hemi_area_fraction",
            "undamaged_area_fraction",
            "left_hemi_undamaged_area_fraction",
            "right_hemi_undamaged_area_fraction",
        ]

        # Filter columns based on which ones are actually available in the expected output
        columns = [col for col in columns if col in expected_output.columns]

        for column in columns:
            with self.subTest(column=column):
                self.assertIn(
                    column, pnt.label_df.columns, f"Missing column in output: {column}"
                )
                self.assertIn(
                    column,
                    expected_output.columns,
                    f"Missing column in expected output: {column}",
                )
                np.testing.assert_array_almost_equal(
                    pnt.label_df[column].values,
                    expected_output[column].values,
                    err_msg=f"Mismatch in column: {column}",
                )

        save_root = os.path.basename(test_case_filename).split(".")[0] + save_suffix
        save_path = os.path.join(
            self.test_case_dir, "..", "demo_data", "outputs", save_root
        )
        # visualisations are optional and can be slow; keep this non-failing and purely informative
        pnt.save_analysis(save_path, create_visualisations=create_visualisations)


test_case_files = [
    "brainglobe_atlas.json",
    "brainglobe_atlas_damage.json",
    "custom_atlas.json",
    "upsized_allen.json",
    "qupath_test.json",
    "empty_segmentation.json",
]


def _make_test_method(
    test_case_file: str,
    *,
    create_visualisations: bool = True,
    save_suffix: str = "",
):
    def _test(self):
        self.run_test_case(
            test_case_file,
            create_visualisations=create_visualisations,
            save_suffix=save_suffix,
        )

    return _test


for test_case_file in test_case_files:
    stem = test_case_file.split(".")[0]
    if stem == "upsized_allen":
        # Run twice to measure the runtime penalty of visualisation generation.
        setattr(
            TestQuantification,
            "test_upsized_allen_without_visualisations",
            _make_test_method(
                test_case_file,
                create_visualisations=False,
                save_suffix="_novis",
            ),
        )
        setattr(
            TestQuantification,
            "test_upsized_allen_with_visualisations",
            _make_test_method(
                test_case_file,
                create_visualisations=True,
                save_suffix="_vis",
            ),
        )
    else:
        setattr(
            TestQuantification,
            f"test_{stem}",
            _make_test_method(test_case_file),
        )

if __name__ == "__main__":
    unittest.main()

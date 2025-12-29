import unittest
import os
import sys
import warnings

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from PyNutil import PyNutil
import json

from timing_utils import TimedTestCase

class TestCellposeQuantification(TimedTestCase):
    def setUp(self):
        self.test_case_dir = os.path.dirname(__file__)
        # Keep test output readable: ignore noisy dependency deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def load_test_case(self, filename):
        test_case_filename = os.path.join(self.test_case_dir, "test_cases", filename)
        with open(test_case_filename, "r") as file:
            return test_case_filename, json.load(file)

    def test_cellpose_quantification(self):
        test_case_filename, test_case = self.load_test_case("cellpose_test.json")
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
            "idx",
            "name",
            "r",
            "g",
            "b",
            "object_count",
            "pixel_count",
            "area_fraction",
        ]

        # Filter columns based on which ones are actually available in the expected output
        columns = [col for col in columns if col in expected_output.columns]

        for column in columns:
            with self.subTest(column=column):
                self.assertIn(
                    column, pnt.label_df.columns, f"Missing column in output: {column}"
                )
                if column in ["idx", "name"]:
                    np.testing.assert_array_equal(
                        pnt.label_df[column].values,
                        expected_output[column].values,
                        err_msg=f"Mismatch in column: {column}",
                    )
                else:
                    np.testing.assert_array_almost_equal(
                        pnt.label_df[column].values,
                        expected_output[column].values,
                        err_msg=f"Mismatch in column: {column}",
                    )

if __name__ == "__main__":
    unittest.main()

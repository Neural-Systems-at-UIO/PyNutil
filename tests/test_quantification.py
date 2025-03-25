import unittest
import os
import sys

sys.path.append(os.path.abspath("/home/harryc/github/PyNutil/"))
import numpy as np
import pandas as pd
from PyNutil import PyNutil
import json


class TestQuantification(unittest.TestCase):
    def setUp(self):
        self.test_case_dir = os.path.dirname(__file__)

    def load_test_case(self, filename):
        return

    def load_test_case(self, filename):
        test_case_filename = os.path.join(self.test_case_dir, "test_cases", filename)
        with open(test_case_filename, "r") as file:
            return test_case_filename, json.load(file)

    def run_test_case(self, test_case_filename):
        test_case_filename, test_case = self.load_test_case(test_case_filename)
        pnt = PyNutil(settings_file=test_case_filename)
        pnt.get_coordinates(object_cutoff=0)
        pnt.quantify_coordinates()
        expected_region_area_path = os.path.join(
            self.test_case_dir,
            test_case["expected_output_folder"],
            "whole_series_report",
            "counts.csv",
        )
        expected_region_area = pd.read_csv(expected_region_area_path, sep=";")
        np.testing.assert_array_almost_equal(
            pnt.label_df["region_area"].values,
            expected_region_area["region_area"].values,
        )
        save_path = os.path.join(self.test_case_dir, "..", "demo_data", "outputs")
        pnt.save_analysis(save_path)


test_case_files = [
    "brainglobe_atlas.json",
    "brainglobe_atlas_damage.json",
    "custom_atlas.json",
]
for test_case_file in test_case_files:

    def test_method(self, test_case_file=test_case_file):
        self.run_test_case(test_case_file)

    setattr(TestQuantification, f'test_{test_case_file.split(".")[0]}', test_method)

if __name__ == "__main__":
    unittest.main()

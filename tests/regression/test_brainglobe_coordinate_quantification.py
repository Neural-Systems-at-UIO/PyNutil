"""Regression tests for BrainGlobe coordinate quantification.

1. Regression: compares counts CSV columns against stored expected output.
2. Consistency: verifies that per-region counts in the meshview JSONs match
   the counts reported in the whole_series_report counts CSV (matched by
   region name, since meshview idx is sequential while report idx is the
   atlas region ID).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PyNutil import PyNutil

from timing_utils import TimedTestCase

TESTS_DIR = os.path.dirname(os.path.dirname(__file__))


class TestBrainGlobeCoordinateQuantification(TimedTestCase):
    """Regression and consistency tests for brainglobe coordinate pipeline."""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        test_case_path = os.path.join(
            TESTS_DIR, "test_cases", "brainglobe_coordinate_test.json"
        )
        with open(test_case_path) as f:
            cls.test_case = json.load(f)

        cls._tmpdir = tempfile.mkdtemp(prefix="pynutil_bg_coord_test_")

        pnt = PyNutil(settings_file=test_case_path)
        pnt.get_coordinates()
        pnt.quantify_coordinates()
        pnt.save_analysis(cls._tmpdir, create_visualisations=False)

        cls.label_df = pnt.label_df.copy()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path):
        with open(path) as f:
            return json.load(f)

    def _input_count(self):
        coord_path = os.path.join(TESTS_DIR, "..", self.test_case["coordinate_file"])
        return len(pd.read_csv(coord_path))

    # ------------------------------------------------------------------
    # 1. Regression: counts CSV matches expected
    # ------------------------------------------------------------------

    def test_counts_regression(self):
        """Counts CSV columns must match the stored expected output."""
        expected_path = os.path.join(
            TESTS_DIR,
            self.test_case["expected_output_folder"],
            "whole_series_report",
            "counts.csv",
        )
        expected = pd.read_csv(expected_path, sep=";")

        columns = [c for c in expected.columns if c in self.label_df.columns]

        for column in columns:
            with self.subTest(column=column):
                if column in ("idx", "name"):
                    np.testing.assert_array_equal(
                        self.label_df[column].values,
                        expected[column].values,
                        err_msg=f"Mismatch in column: {column}",
                    )
                else:
                    np.testing.assert_array_almost_equal(
                        self.label_df[column].values,
                        expected[column].values,
                        err_msg=f"Mismatch in column: {column}",
                    )

    # ------------------------------------------------------------------
    # 2. Consistency: meshview per-region counts match report (by name)
    # ------------------------------------------------------------------

    def _meshview_counts_by_name(self, filename):
        meshview_path = os.path.join(
            self._tmpdir, "whole_series_meshview", filename
        )
        meshview = self._load_json(meshview_path)
        return {entry["name"]: entry["count"] for entry in meshview}

    def test_meshview_object_counts_match_report(self):
        """Per-region object counts in objects_meshview.json must match
        the object_count column in the counts CSV (matched by name)."""
        mv_counts = self._meshview_counts_by_name("objects_meshview.json")

        for _, row in self.label_df.iterrows():
            name = row["name"]
            report_count = int(row["object_count"])
            mv_count = mv_counts.get(name, 0)
            if report_count == 0 and name not in mv_counts:
                continue
            with self.subTest(name=name):
                self.assertEqual(
                    mv_count,
                    report_count,
                    f"'{name}': meshview={mv_count} != report={report_count}",
                )

    def test_meshview_pixel_counts_match_report(self):
        """Per-region pixel counts in pixels_meshview.json must match
        the pixel_count column in the counts CSV (matched by name)."""
        mv_counts = self._meshview_counts_by_name("pixels_meshview.json")

        for _, row in self.label_df.iterrows():
            name = row["name"]
            report_count = int(row["pixel_count"])
            mv_count = mv_counts.get(name, 0)
            if report_count == 0 and name not in mv_counts:
                continue
            with self.subTest(name=name):
                self.assertEqual(
                    mv_count,
                    report_count,
                    f"'{name}': meshview={mv_count} != report={report_count}",
                )

    # ------------------------------------------------------------------
    # 3. Internal consistency: meshview triplet lengths
    # ------------------------------------------------------------------

    def test_meshview_triplet_lengths_match_counts(self):
        """Each meshview entry's triplets array length must equal 3 * count."""
        for filename in ("objects_meshview.json", "pixels_meshview.json"):
            meshview_path = os.path.join(
                self._tmpdir, "whole_series_meshview", filename
            )
            meshview = self._load_json(meshview_path)
            for entry in meshview:
                with self.subTest(file=filename, name=entry["name"]):
                    expected_len = entry["count"] * 3
                    self.assertEqual(
                        len(entry["triplets"]),
                        expected_len,
                        f"{filename} '{entry['name']}': triplets length "
                        f"{len(entry['triplets'])} != 3 * count ({expected_len})",
                    )

    # ------------------------------------------------------------------
    # 4. Totals match input count
    # ------------------------------------------------------------------

    def test_report_total_matches_input(self):
        """Sum of object_count in the report must equal input row count."""
        input_count = self._input_count()
        total = int(self.label_df["object_count"].sum())
        self.assertEqual(
            total,
            input_count,
            f"Report object_count total {total} != input CSV rows {input_count}",
        )

    def test_meshview_total_matches_input(self):
        """Total points across all meshview entries must equal the number
        of input coordinates."""
        input_count = self._input_count()

        for filename in ("objects_meshview.json", "pixels_meshview.json"):
            with self.subTest(file=filename):
                meshview_path = os.path.join(
                    self._tmpdir, "whole_series_meshview", filename
                )
                meshview = self._load_json(meshview_path)
                total = sum(entry["count"] for entry in meshview)
                self.assertEqual(
                    total,
                    input_count,
                    f"{filename} total {total} != input CSV rows {input_count}",
                )

    # ------------------------------------------------------------------
    # 5. Hemisphere consistency
    # ------------------------------------------------------------------

    def test_report_hemi_lte_total(self):
        """left_hemi + right_hemi must be <= total for each region."""
        if "left_hemi_object_count" not in self.label_df.columns:
            self.skipTest("No hemisphere columns in output")

        for _, row in self.label_df.iterrows():
            with self.subTest(name=row["name"]):
                hemi_sum = (
                    row["left_hemi_object_count"] + row["right_hemi_object_count"]
                )
                self.assertLessEqual(
                    hemi_sum,
                    row["object_count"],
                    f"left+right ({hemi_sum}) > total ({row['object_count']})",
                )

    def test_hemisphere_meshview_totals_sum(self):
        """Left + right hemisphere meshview totals must be <= full total."""
        for kind in ("objects", "pixels"):
            with self.subTest(kind=kind):
                full_path = os.path.join(
                    self._tmpdir, "whole_series_meshview",
                    f"{kind}_meshview.json",
                )
                left_path = os.path.join(
                    self._tmpdir, "whole_series_meshview",
                    f"left_hemisphere_{kind}_meshview.json",
                )
                right_path = os.path.join(
                    self._tmpdir, "whole_series_meshview",
                    f"right_hemisphere_{kind}_meshview.json",
                )

                full_total = sum(e["count"] for e in self._load_json(full_path))
                left_total = sum(e["count"] for e in self._load_json(left_path))
                right_total = sum(e["count"] for e in self._load_json(right_path))

                self.assertLessEqual(
                    left_total + right_total,
                    full_total,
                    f"left ({left_total}) + right ({right_total}) > full ({full_total})",
                )


if __name__ == "__main__":
    unittest.main()

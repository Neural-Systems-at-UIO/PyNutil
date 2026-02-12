"""Regression tests for brainglobe-registration intensity MeshView output.

Compares the meshview JSON files produced by the intensity pipeline
against validated expected outputs stored in
tests/expected_outputs/brainglobe_registration_intensity/.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyNutil import PyNutil

from timing_utils import TimedTestCase


class TestBrainGlobeMeshviewRegression(TimedTestCase):
    """Verify that brainglobe intensity MeshView JSON output matches reference."""

    EXPECTED_ROOT = os.path.join(
        os.path.dirname(__file__),
        "expected_outputs",
        "brainglobe_registration_intensity",
    )
    MESHVIEW_DIR = "whole_series_meshview"

    MESHVIEW_FILES = [
        "pixels_meshview.json",
        "left_hemisphere_pixels_meshview.json",
        "right_hemisphere_pixels_meshview.json",
    ]

    @classmethod
    def setUpClass(cls):
        """Run the intensity pipeline once and save results to a temp dir."""
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        test_case_path = os.path.join(
            os.path.dirname(__file__),
            "test_cases",
            "brainglobe_registration_intensity.json",
        )
        cls._tmpdir = tempfile.mkdtemp(prefix="pynutil_bg_meshview_test_")

        pnt = PyNutil(settings_file=test_case_path)
        pnt.get_coordinates()
        pnt.quantify_coordinates()
        pnt.save_analysis(cls._tmpdir, create_visualisations=False)

    @classmethod
    def tearDownClass(cls):
        """Remove the temp directory."""
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _triplets_to_points(triplets):
        """Convert a flat [x,y,z,x,y,z,...] list to a sorted (N,3) array."""
        pts = np.array(triplets, dtype=np.float64).reshape(-1, 3)
        order = np.lexsort(pts[:, ::-1].T)
        return pts[order]

    def _assert_meshview_equal(self, expected_path, actual_path):
        """Compare two MeshView JSON files region-by-region."""
        expected = self._load_json(expected_path)
        actual = self._load_json(actual_path)

        exp_by_name = {e["name"]: e for e in expected}
        act_by_name = {a["name"]: a for a in actual}

        self.assertEqual(
            set(exp_by_name.keys()),
            set(act_by_name.keys()),
            "Region name sets differ",
        )

        for name in exp_by_name:
            with self.subTest(region=name):
                exp_entry = exp_by_name[name]
                act_entry = act_by_name[name]

                self.assertEqual(
                    exp_entry["count"],
                    act_entry["count"],
                    f"Region '{name}': count mismatch",
                )

                exp_pts = self._triplets_to_points(exp_entry["triplets"])
                act_pts = self._triplets_to_points(act_entry["triplets"])

                self.assertEqual(
                    exp_pts.shape,
                    act_pts.shape,
                    f"Region '{name}': point array shape mismatch",
                )
                np.testing.assert_allclose(
                    exp_pts,
                    act_pts,
                    atol=1e-6,
                    err_msg=f"Region '{name}': point coordinates differ",
                )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_meshview_files_match_expected(self):
        """Each whole_series_meshview JSON must match the expected reference."""
        for filename in self.MESHVIEW_FILES:
            with self.subTest(file=filename):
                expected_path = os.path.join(
                    self.EXPECTED_ROOT, self.MESHVIEW_DIR, filename
                )
                actual_path = os.path.join(
                    self._tmpdir, self.MESHVIEW_DIR, filename
                )

                self.assertTrue(
                    os.path.exists(expected_path),
                    f"Expected file missing: {expected_path}",
                )
                self.assertTrue(
                    os.path.exists(actual_path),
                    f"Actual output file missing: {actual_path}",
                )

                self._assert_meshview_equal(expected_path, actual_path)


if __name__ == "__main__":
    unittest.main()

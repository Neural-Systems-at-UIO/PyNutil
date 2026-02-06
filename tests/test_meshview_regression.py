"""Regression tests for MeshView JSON output.

Compares the meshview JSON files produced by save_analysis against
validated expected outputs stored in tests/expected_outputs/.
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


class TestMeshviewRegression(TimedTestCase):
    """Verify that MeshView JSON output matches validated reference files.

    The comparison is order-independent: regions are matched by name and
    points within each region are compared as unordered sets of 3D
    coordinates (tolerance 1e-6).
    """

    EXPECTED_ROOT = os.path.join(
        os.path.dirname(__file__), "expected_outputs", "brainglobe_atlas"
    )
    MESHVIEW_DIR = "whole_series_meshview"

    MESHVIEW_FILES = [
        "objects_meshview.json",
        "pixels_meshview.json",
        "left_hemisphere_objects_meshview.json",
        "left_hemisphere_pixels_meshview.json",
        "right_hemisphere_objects_meshview.json",
        "right_hemisphere_pixels_meshview.json",
    ]

    @classmethod
    def setUpClass(cls):
        """Run the pipeline once and save results to a temp directory."""
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        test_case_path = os.path.join(
            os.path.dirname(__file__), "test_cases", "brainglobe_atlas.json"
        )
        cls._tmpdir = tempfile.mkdtemp(prefix="pynutil_meshview_test_")

        pnt = PyNutil(settings_file=test_case_path)
        pnt.get_coordinates(object_cutoff=0)
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
        # Sort rows lexicographically so order doesn't matter
        order = np.lexsort(pts[:, ::-1].T)
        return pts[order]

    def _assert_meshview_equal(self, expected_path, actual_path):
        """Compare two MeshView JSON files region-by-region, order-independent."""
        expected = self._load_json(expected_path)
        actual = self._load_json(actual_path)

        # Build lookup by region name for both sides
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

                # Point count must match
                self.assertEqual(
                    exp_entry["count"],
                    act_entry["count"],
                    f"Region '{name}': count mismatch",
                )
                # Color must match
                for key in ("r", "g", "b"):
                    self.assertEqual(
                        exp_entry[key],
                        act_entry[key],
                        f"Region '{name}': '{key}' mismatch",
                    )

                # Compare 3D points as unordered sets (sorted lexicographically)
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
    # Tests – one subtest per meshview file
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

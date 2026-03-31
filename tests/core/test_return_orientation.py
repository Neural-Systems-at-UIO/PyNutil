"""Test that return_orientation produces consistent internal coordinates."""

import os
import unittest

import numpy as np

from brainglobe_atlasapi import BrainGlobeAtlas
from PyNutil import seg_to_coords, read_alignment, read_segmentation_dir
from PyNutil.io.atlas_loader import resolve_atlas


TEST_DIR = os.path.dirname(os.path.dirname(__file__))


class TestReturnOrientation(unittest.TestCase):
    """Verify that different return_orientation values yield the same
    internal (lpi) coordinates after round-tripping."""

    @classmethod
    def setUpClass(cls):
        alignment_json = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "alignment.json"
        )
        seg_folder = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "segmentations"
        )
        if not os.path.isfile(alignment_json):
            raise unittest.SkipTest("Test data not found")

        cls.atlas = resolve_atlas(BrainGlobeAtlas("allen_mouse_25um"))
        cls.alignment = read_alignment(alignment_json)
        cls.image_series = read_segmentation_dir(seg_folder)

    def _run(self, orientation):
        return seg_to_coords(
            self.image_series,
            self.alignment,
            self.atlas,
            return_orientation=orientation,
        )

    def test_orientations_produce_same_internal_points(self):
        """Points reoriented back to lpi should match regardless of return_orientation."""
        result_lpi = self._run("lpi")
        result_asr = self._run("asr")
        result_ras = self._run("ras")

        lpi_pts = result_lpi.points.points_in_internal_orientation(
            result_lpi.points.points
        )
        asr_pts = result_asr.points.points_in_internal_orientation(
            result_asr.points.points
        )
        ras_pts = result_ras.points.points_in_internal_orientation(
            result_ras.points.points
        )

        np.testing.assert_allclose(lpi_pts, asr_pts, atol=1e-6)
        np.testing.assert_allclose(lpi_pts, ras_pts, atol=1e-6)

    def test_orientation_stored_on_result(self):
        """The orientation field should reflect the requested orientation."""
        result = self._run("asr")
        self.assertEqual(result.points.orientation, "asr")
        self.assertEqual(result.objects.orientation, "asr")

    def test_different_orientations_have_different_raw_points(self):
        """Raw points should differ when orientations differ."""
        result_lpi = self._run("lpi")
        result_asr = self._run("asr")

        # Points should not be identical (different coordinate systems)
        self.assertFalse(
            np.allclose(result_lpi.points.points, result_asr.points.points),
            "Raw points should differ between lpi and asr orientations",
        )

    def test_invalid_orientation_has_helpful_error(self):
        """Invalid orientation codes should fail with a brief explanation."""
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid orientation code 'lsr'.*left/right.*superior/inferior.*anterior/posterior",
        ):
            self._run("lsr")


if __name__ == "__main__":
    unittest.main()

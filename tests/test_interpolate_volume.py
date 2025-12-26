import os
import unittest

import numpy as np
import nrrd

from PyNutil import PyNutil
from timing_utils import TimedTestCase


class TestInterpolateVolume(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(__file__)
        self.settings_path = os.path.join(self.tests_dir, "test_cases", "brainglobe_atlas.json")
        self.expected_dir = os.path.join(
            self.tests_dir, "expected_outputs"
        )
        self.save_root = os.path.basename(self.settings_path).split(".")[0] + "_interp_k5"
        self.output_dir = os.path.join(self.tests_dir, "..", "demo_data", "outputs", self.save_root)
        self.output_report_dir = os.path.join(self.output_dir, "interpolated_volume")

        self.expected_case_dir = os.path.join(self.expected_dir, self.save_root)
        self.expected_report_dir = os.path.join(self.expected_case_dir, "interpolated_volume")

    def _generate_outputs(self):
        # SciPy is required for kNN interpolation.
        try:
            import scipy  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"SciPy is required for this test: {exc}")

        pnt = PyNutil(settings_file=self.settings_path)
        pnt.get_coordinates(object_cutoff=0)
        pnt.quantify_coordinates()

        # Downscale to keep runtime/memory small and stable.
        sx, sy, sz = (int(x) for x in pnt.atlas_volume.shape)
        max_dim = max(sx, sy, sz)
        # Target max dimension ~80 voxels.
        scale = min(80.0 / max_dim, 1.0)
        shape = (
            max(10, int(round(sx * scale))),
            max(10, int(round(sy * scale))),
            max(10, int(round(sz * scale))),
        )

        # Faithful plane-based volume: every pixel in each section plane contributes
        # (0 for background, 1 for segmentation colour), and fv counts coverage.
        pnt.build_volume_from_sections(
            resolution_scale=scale,
            shape=shape,
            missing_fill=np.nan,
            do_interpolation=True,
            k=5,
            weights="uniform",
            use_atlas_mask=True,
            non_linear=True,
        )

        # Volumes should be written as NRRD during save_analysis.
        pnt.save_analysis(self.output_dir, create_visualisations=False)

    def test_interpolate_volume_k5_matches_expected(self):
        self._generate_outputs()

        got_interp_path = os.path.join(self.output_report_dir, "interpolated_volume.nrrd")
        got_freq_path = os.path.join(self.output_report_dir, "frequency_volume.nrrd")
        self.assertTrue(os.path.exists(got_interp_path), f"Missing output: {got_interp_path}")
        self.assertTrue(os.path.exists(got_freq_path), f"Missing output: {got_freq_path}")

        exp_interp_path = os.path.join(self.expected_report_dir, "interpolated_volume.nrrd")
        exp_freq_path = os.path.join(self.expected_report_dir, "frequency_volume.nrrd")

        if not (os.path.exists(exp_interp_path) and os.path.exists(exp_freq_path)):
            self.fail(
                "Expected interpolate_volume NRRD outputs were missing. "
                f"Generated candidate outputs under: {self.output_report_dir}. "
                f"Please verify and copy that folder to: {self.expected_report_dir}, then re-run the tests."
            )

        got_interp, _ = nrrd.read(got_interp_path)
        got_freq, _ = nrrd.read(got_freq_path)
        exp_interp, _ = nrrd.read(exp_interp_path)
        exp_freq, _ = nrrd.read(exp_freq_path)

        self.assertEqual(exp_interp.shape, got_interp.shape)
        self.assertEqual(exp_freq.shape, got_freq.shape)
        np.testing.assert_allclose(exp_interp, got_interp, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(exp_freq, got_freq)


if __name__ == "__main__":
    unittest.main()

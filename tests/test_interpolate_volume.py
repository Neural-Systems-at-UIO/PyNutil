import os
import shutil
import tempfile
import unittest

import numpy as np
import nibabel as nib

from PyNutil import PyNutil
try:
    # When run via `python -m unittest discover` from repo root
    from tests.timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    # When run with tests/ on sys.path
    from timing_utils import TimedTestCase


class TestInterpolateVolume(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(__file__)
        self.settings_path = os.path.join(self.tests_dir, "test_cases", "brainglobe_atlas.json")
        self.expected_dir = os.path.join(
            self.tests_dir, "expected_outputs"
        )
        self.save_root = os.path.basename(self.settings_path).split(".")[0] + "_interp_k5"
        # Note: demo_data outputs are for optional human inspection only.
        # Tests must never read from demo_data.
        self.demo_output_dir = os.path.join(
            self.tests_dir, "..", "demo_data", "outputs", self.save_root
        )
        self.demo_report_dir = os.path.join(self.demo_output_dir, "interpolated_volume")

        self.expected_case_dir = os.path.join(self.expected_dir, self.save_root)
        self.expected_report_dir = os.path.join(self.expected_case_dir, "interpolated_volume")

    def _generate_outputs(self, output_dir: str):
        pnt = PyNutil(settings_file=self.settings_path)
        pnt.get_coordinates(object_cutoff=0)
        pnt.quantify_coordinates()

        # Downscale to keep runtime/memory small and stable.
        sx, sy, sz = (int(x) for x in pnt.atlas_volume.shape)
        max_dim = max(sx, sy, sz)
        # Target max dimension ~80 voxels.
        scale = min(80.0 / max_dim, 1.0)

        # Faithful plane-based volume: every pixel in each section plane contributes
        # (0 for background, 1 for segmentation colour), and fv counts coverage.
        pnt.build_volume_from_sections(
            scale=scale,
            missing_fill=np.nan,
            do_interpolation=True,
            k=5,
            weights="uniform",
            use_atlas_mask=True,
            non_linear=True,
        )

        # Volumes should be written as NII during save_analysis.
        pnt.save_analysis(output_dir, create_visualisations=True)

    def test_interpolate_volume_k5_matches_expected(self):
        with tempfile.TemporaryDirectory(prefix="pynutil_interp_k5_") as tmpdir:
            output_dir = os.path.join(tmpdir, self.save_root)
            output_report_dir = os.path.join(output_dir, "interpolated_volume")

            self._generate_outputs(output_dir)

            got_interp_path = os.path.join(
                output_report_dir, "interpolated_volume.nii.gz"
            )
            got_freq_path = os.path.join(output_report_dir, "frequency_volume.nii.gz")
            self.assertTrue(os.path.exists(got_interp_path), f"Missing output: {got_interp_path}")
            self.assertTrue(os.path.exists(got_freq_path), f"Missing output: {got_freq_path}")

            exp_interp_path = os.path.join(
                self.expected_report_dir, "interpolated_volume.nii.gz"
            )
            exp_freq_path = os.path.join(self.expected_report_dir, "frequency_volume.nii.gz")

            if not (os.path.exists(exp_interp_path) and os.path.exists(exp_freq_path)):
                # Copy outputs for human inspection (never used for assertions).
                os.makedirs(os.path.dirname(self.demo_report_dir), exist_ok=True)
                shutil.rmtree(self.demo_output_dir, ignore_errors=True)
                shutil.copytree(output_dir, self.demo_output_dir)

                self.fail(
                    "Expected interpolate_volume NIfTI outputs were missing. "
                    f"Generated candidate outputs have been copied to: {self.demo_report_dir}. "
                    f"Please verify and copy that folder to: {self.expected_report_dir}, then re-run the tests."
                )

            # Optional: also persist outputs to demo_data for inspection.
            if os.environ.get("PYNUTIL_SAVE_TEST_OUTPUTS", "").strip() in {"1", "true", "True"}:
                os.makedirs(os.path.dirname(self.demo_report_dir), exist_ok=True)
                shutil.rmtree(self.demo_output_dir, ignore_errors=True)
                shutil.copytree(output_dir, self.demo_output_dir)

            # Use np.asarray (not np.asanyarray) to avoid NumPy's copy=False pathway,
            # which can produce empty arrays for nibabel's ArrayProxy on newer NumPy.
            got_interp = np.asarray(nib.load(got_interp_path).dataobj)
            got_freq = np.asarray(nib.load(got_freq_path).dataobj)
            exp_interp = np.asarray(nib.load(exp_interp_path).dataobj)
            exp_freq = np.asarray(nib.load(exp_freq_path).dataobj)

            self.assertEqual(exp_interp.shape, got_interp.shape)
            self.assertEqual(exp_freq.shape, got_freq.shape)
            # Volumes are saved as uint8; compare exactly.
            np.testing.assert_array_equal(exp_interp, got_interp)
            np.testing.assert_array_equal(exp_freq, got_freq)


if __name__ == "__main__":
    unittest.main()

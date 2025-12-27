import os
import shutil
import tempfile
import unittest

import numpy as np
import nibabel as nib

from PyNutil import PyNutil
from PyNutil.io.volume_nifti import scale_to_uint8
from tests.test_helpers import copy_tree_to_demo, small_volume_scale
try:
    # When run via `python -m unittest discover` from repo root
    from tests.timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    # When run with tests/ on sys.path
    from timing_utils import TimedTestCase


class TestBuildVolumeFromSections(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(__file__)
        self.settings_path = os.path.join(self.tests_dir, "test_cases", "brainglobe_atlas.json")
        self.expected_dir = os.path.join(
            self.tests_dir, "expected_outputs"
        )
        # Keep the expected-output directory name stable to avoid re-generating
        # large binary fixtures.
        self.save_root = os.path.basename(self.settings_path).split(".")[0] + "_interp_k5"
        # Note: demo_data outputs are for optional human inspection only.
        # Tests must never read from demo_data.
        self.demo_output_dir = os.path.join(
            self.tests_dir, "..", "demo_data", "outputs", self.save_root
        )
        self.demo_report_dir = os.path.join(self.demo_output_dir, "interpolated_volume")

        self.expected_case_dir = os.path.join(self.expected_dir, self.save_root)
        self.expected_report_dir = os.path.join(self.expected_case_dir, "interpolated_volume")

    def _generate_pnt(self):
        pnt = PyNutil(settings_file=self.settings_path)
        pnt.get_coordinates(object_cutoff=0)
        pnt.quantify_coordinates()

        # Downscale to keep runtime/memory small and stable.
        scale = small_volume_scale(pnt.atlas_volume.shape)

        # Plane-based volume: every pixel in each section plane contributes
        # (0 for background, 1 for segmentation colour), and fv counts coverage.
        pnt.interpolate_volume(
            scale=scale,
            missing_fill=np.nan,
            do_interpolation=True,
            k=5,
            use_atlas_mask=True,
            non_linear=True,
        )
        return pnt

    def test_interpolate_volume_k5_matches_expected(self):
        pnt = self._generate_pnt()

        # Volumes are saved as uint8 in NIfTI; we scale them here to match
        # the expected disk fixtures without having to read back what we just wrote.
        got_interp = scale_to_uint8(pnt.interpolated_volume)
        got_freq = scale_to_uint8(pnt.frequency_volume)

        exp_interp_path = os.path.join(
            self.expected_report_dir, "interpolated_volume.nii.gz"
        )
        exp_freq_path = os.path.join(self.expected_report_dir, "frequency_volume.nii.gz")

        with tempfile.TemporaryDirectory(prefix="pynutil_build_from_sections_k5_") as tmpdir:
            output_dir = os.path.join(tmpdir, self.save_root)
            # Volumes should be written as NII during save_analysis for human inspection.
            pnt.save_analysis(output_dir, create_visualisations=True)

            if not (os.path.exists(exp_interp_path) and os.path.exists(exp_freq_path)):
                # Copy outputs for human inspection (never used for assertions).
                copy_tree_to_demo(output_dir, self.demo_output_dir)

                self.fail(
                    "Expected volume NIfTI outputs were missing. "
                    f"Generated candidate outputs have been copied to: {self.demo_report_dir}. "
                    f"Please verify and copy that folder to: {self.expected_report_dir}, then re-run the tests."
                )

            # Always persist outputs to demo_data for human inspection.
            # Tests must never read from demo_data.
            copy_tree_to_demo(output_dir, self.demo_output_dir)

        # Use np.asarray (not np.asanyarray) to avoid NumPy's copy=False pathway,
        # which can produce empty arrays for nibabel's ArrayProxy on newer NumPy.
        exp_interp = np.asarray(nib.load(exp_interp_path).dataobj)
        exp_freq = np.asarray(nib.load(exp_freq_path).dataobj)

        self.assertEqual(exp_interp.shape, got_interp.shape)
        self.assertEqual(exp_freq.shape, got_freq.shape)
        # Compare exactly against the expected uint8 volumes.
        self.assertTrue(np.array_equal(exp_interp, got_interp))
        self.assertTrue(np.array_equal(exp_freq, got_freq))


if __name__ == "__main__":
    unittest.main()

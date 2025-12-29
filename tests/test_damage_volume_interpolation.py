import os
import unittest
import numpy as np
from PyNutil import PyNutil
from tests.test_helpers import make_pynutil_ready, small_volume_scale

try:
    # When run via `python -m unittest discover` from repo root
    from tests.timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    # When run with tests/ on sys.path
    from timing_utils import TimedTestCase

class TestDamageVolumeInterpolation(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(__file__)
        self.settings_path = os.path.join(
            self.tests_dir, "test_cases", "brainglobe_atlas_damage.json"
        )

    def test_damage_volume_creation(self):
        # Initialize PyNutil with the damage test case
        pnt = make_pynutil_ready(self.settings_path)

        # Use a small scale for faster testing
        scale = small_volume_scale(pnt.atlas_volume.shape)

        # Run interpolation
        pnt.interpolate_volume(
            scale=scale,
            do_interpolation=False,
            non_linear=False
        )

        # Check if damage_volume exists and has the correct shape
        self.assertTrue(hasattr(pnt, "damage_volume"), "PyNutil instance should have damage_volume attribute")
        self.assertEqual(pnt.damage_volume.shape, pnt.interpolated_volume.shape, "Damage volume shape should match interpolated volume shape")

        # Check if damage_volume contains non-zero values
        # The brainglobe_atlas_damage test case is known to have damage markers
        self.assertTrue(np.any(pnt.damage_volume > 0), "Damage volume should contain non-zero values for this test case")

        # Check if damage_volume is binary (0 or 1)
        unique_values = np.unique(pnt.damage_volume)
        for val in unique_values:
            self.assertIn(val, [0, 1], f"Damage volume should only contain 0 or 1, found {val}")

    def test_damage_volume_interpolation(self):
        # Initialize PyNutil with the damage test case
        pnt = make_pynutil_ready(self.settings_path)

        # Use a small scale for faster testing
        scale = small_volume_scale(pnt.atlas_volume.shape)

        # Run without interpolation first to get baseline
        pnt.interpolate_volume(
            scale=scale,
            do_interpolation=False,
            non_linear=False
        )
        dv_no_interp = pnt.damage_volume.copy()
        count_no_interp = np.sum(dv_no_interp > 0)

        # Run with interpolation
        pnt.interpolate_volume(
            scale=scale,
            do_interpolation=True,
            k=5,
            non_linear=False
        )
        dv_interp = pnt.damage_volume.copy()
        count_interp = np.sum(dv_interp > 0)

        # With interpolation, the damage volume should generally have more (or equal) damaged voxels
        # because it's a "max" aggregation over k neighbors.
        self.assertGreaterEqual(count_interp, count_no_interp, "Interpolated damage volume should have at least as many damaged voxels as non-interpolated")

        # Verify that it's still binary
        unique_values = np.unique(dv_interp)
        for val in unique_values:
            self.assertIn(val, [0, 1], f"Interpolated damage volume should only contain 0 or 1, found {val}")

    def test_damage_volume_persistence(self):
        import tempfile
        import nibabel as nib

        pnt = make_pynutil_ready(self.settings_path)
        scale = small_volume_scale(pnt.atlas_volume.shape)

        pnt.interpolate_volume(
            scale=scale,
            do_interpolation=True,
            k=5,
            non_linear=False
        )
        demo_output_dir = os.path.join(
            self.tests_dir, "..", "demo_data", "outputs", "interpolated_damage"
        )
        pnt.save_analysis(demo_output_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            pnt.save_analysis(tmpdir)

            damage_nifti_path = os.path.join(tmpdir, "interpolated_volume", "damage_volume.nii.gz")
            self.assertTrue(os.path.exists(damage_nifti_path), f"Damage volume NIfTI should be saved at {damage_nifti_path}")

            # Load and verify the saved NIfTI
            img = nib.load(damage_nifti_path)
            data = img.get_fdata()
            self.assertTrue(np.any(data > 0), "Saved damage volume should contain non-zero values")
            # NIfTI saving scales to uint8 (0-255)
            self.assertEqual(np.max(data), 255.0, "Max value in saved damage volume should be 255 (scaled from 1)")

if __name__ == "__main__":
    unittest.main()

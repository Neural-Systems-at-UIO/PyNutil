import json
import os
import unittest
import numpy as np
from PyNutil import interpolate_volume, save_analysis, read_segmentation_dir
from test_helpers import run_pipeline_from_settings_file, small_volume_scale, load_atlas_from_settings

try:
    # When run via `python -m unittest discover` from repo root
    from timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    # When run with tests/ on sys.path
    from timing_utils import TimedTestCase

class TestDamageVolumeInterpolation(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(os.path.dirname(__file__))
        self.settings_path = os.path.join(
            self.tests_dir, "test_cases", "brainglobe_atlas_damage.json"
        )
        with open(self.settings_path) as f:
            self.settings = json.load(f)

    def _run_interpolation(self, do_interpolation=False, k=5):
        atlas, result, label_df, alignment = run_pipeline_from_settings_file(self.settings_path)
        scale = small_volume_scale(atlas.volume.shape)

        image_series = read_segmentation_dir(
            self.settings["segmentation_folder"],
            pixel_id=self.settings.get("colour", [0, 0, 0]),
        )
        volumes = interpolate_volume(
            image_series=image_series,
            registration=alignment,
            colour=self.settings.get("colour", [0, 0, 0]),
            atlas=atlas,
            scale=scale,
            do_interpolation=do_interpolation,
            k=k,
        )
        return atlas, result, label_df, volumes

    def test_damage_volume_creation(self):
        atlas, result, label_df, volumes = self._run_interpolation(
            do_interpolation=False,
        )

        # Check if damage_volume has the correct shape
        self.assertEqual(volumes.damage.shape, volumes.value.shape, "Damage volume shape should match interpolated volume shape")

        # Check if damage_volume contains non-zero values
        # The brainglobe_atlas_damage test case is known to have damage markers
        self.assertTrue(np.any(volumes.damage > 0), "Damage volume should contain non-zero values for this test case")

        # Check if damage_volume is binary (0 or 1)
        unique_values = np.unique(volumes.damage)
        for val in unique_values:
            self.assertIn(val, [0, 1], f"Damage volume should only contain 0 or 1, found {val}")

    def test_damage_volume_interpolation(self):
        # Run without interpolation first to get baseline
        _, _, _, volumes_no_interp = self._run_interpolation(
            do_interpolation=False,
        )
        count_no_interp = np.sum(volumes_no_interp.damage > 0)

        # Run with interpolation
        _, _, _, volumes_interp = self._run_interpolation(
            do_interpolation=True, k=5,
        )
        count_interp = np.sum(volumes_interp.damage > 0)

        # With interpolation, the damage volume should generally have more (or equal) damaged voxels
        # because it's a "max" aggregation over k neighbors.
        self.assertGreaterEqual(count_interp, count_no_interp, "Interpolated damage volume should have at least as many damaged voxels as non-interpolated")

        # Verify that it's still binary
        unique_values = np.unique(volumes_interp.damage)
        for val in unique_values:
            self.assertIn(val, [0, 1], f"Interpolated damage volume should only contain 0 or 1, found {val}")

    def test_damage_volume_persistence(self):
        import tempfile
        import nibabel as nib

        atlas, result, label_df, volumes = self._run_interpolation(
            do_interpolation=True, k=5,
        )

        demo_output_dir = os.path.join(
            self.tests_dir, "..", "demo_data", "outputs", "interpolated_damage"
        )
        save_analysis(demo_output_dir, result, atlas, label_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_analysis(tmpdir, result, atlas, label_df)

            damage_nifti_path = os.path.join(tmpdir, "interpolated_volume", "damage_volume.nii.gz")
            # Note: save_analysis does not save volume niftis; the damage volume
            # persistence test needs save_volume_niftis for that.
            from PyNutil import save_volume_niftis
            save_volume_niftis(
                output_folder=tmpdir,
                volumes=volumes,
                atlas_volume=atlas.volume,
                voxel_size_um=atlas.voxel_size_um,
            )

            self.assertTrue(os.path.exists(damage_nifti_path), f"Damage volume NIfTI should be saved at {damage_nifti_path}")

            # Load and verify the saved NIfTI
            img = nib.load(damage_nifti_path)
            data = img.get_fdata()
            self.assertTrue(np.any(data > 0), "Saved damage volume should contain non-zero values")
            # NIfTI saving scales to uint8 (0-255)
            self.assertEqual(np.max(data), 255.0, "Max value in saved damage volume should be 255 (scaled from 1)")

if __name__ == "__main__":
    unittest.main()

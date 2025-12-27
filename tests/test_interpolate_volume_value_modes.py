import os
import shutil
import tempfile
import unittest

import numpy as np

from PyNutil import PyNutil
from tests.test_helpers import copy_tree_to_demo, make_pynutil_ready, small_volume_scale

try:
    # When run via `python -m unittest discover` from repo root
    from tests.timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    # When run with tests/ on sys.path
    from timing_utils import TimedTestCase


class TestInterpolateVolumeValueModes(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(__file__)
        self.settings_path = os.path.join(
            self.tests_dir, "test_cases", "brainglobe_atlas.json"
        )

        self.demo_outputs_root = os.path.join(
            self.tests_dir, "..", "demo_data", "outputs", "interpolate_value_modes"
        )

    def _scale_for_small_volume(self, atlas_shape):
        return small_volume_scale(atlas_shape)

    def test_value_mode_mean_matches_pixel_count_over_frequency(self):
        pnt = make_pynutil_ready(self.settings_path)

        scale = self._scale_for_small_volume(pnt.atlas_volume.shape)

        pnt.interpolate_volume(
            scale=scale,
            missing_fill=0.0,
            do_interpolation=False,
            non_linear=False,
            value_mode="pixel_count",
        )
        gv_pc, fv = pnt.interpolated_volume, pnt.frequency_volume

        pnt.interpolate_volume(
            scale=scale,
            missing_fill=-1.0,
            do_interpolation=False,
            non_linear=False,
            value_mode="mean",
        )
        gv_mean, fv2 = pnt.interpolated_volume, pnt.frequency_volume
        with tempfile.TemporaryDirectory(prefix="pynutil_interpolate_value_modes_mean_") as tmpdir:
            out_root = os.path.join(tmpdir, "mean_vs_pixel_count")

            # Save pixel_count
            pnt.interpolate_volume(
                scale=scale,
                missing_fill=0.0,
                do_interpolation=False,
                non_linear=False,
                value_mode="pixel_count",
            )
            pnt.save_analysis(os.path.join(out_root, "pixel_count"), create_visualisations=True)

            # Save mean
            pnt.interpolate_volume(
                scale=scale,
                missing_fill=-1.0,
                do_interpolation=False,
                non_linear=False,
                value_mode="mean",
            )
            pnt.save_analysis(os.path.join(out_root, "mean"), create_visualisations=True)

            copy_tree_to_demo(
                out_root,
                os.path.join(self.demo_outputs_root, "mean_vs_pixel_count"),
            )

        # NOTE: NumPy 2.2.6 on Python 3.14 has a regression where
        # np.testing.assert_array_equal can fail even for equal arrays.
        self.assertTrue(np.array_equal(fv, fv2))

        expected = np.full(gv_pc.shape, -1.0, dtype=np.float32)
        covered = fv != 0
        expected[covered] = gv_pc[covered] / fv[covered].astype(np.float32)

        # NOTE: NumPy 2.2.6 on Python 3.14 has a regression where
        # np.testing.assert_allclose can fail even for equal arrays.
        self.assertTrue(np.allclose(gv_mean, expected, rtol=0, atol=1e-6))

        # Sanity: segmented pixel count is always <= contributing pixel count.
        self.assertTrue(np.all(gv_pc <= fv.astype(np.float32)))

    def test_value_mode_object_count_basic_invariants(self):
        pnt = make_pynutil_ready(self.settings_path)

        scale = self._scale_for_small_volume(pnt.atlas_volume.shape)

        pnt.interpolate_volume(
            scale=scale,
            missing_fill=0.0,
            do_interpolation=False,
            non_linear=False,
            value_mode="pixel_count",
        )
        gv_pc, fv = pnt.interpolated_volume, pnt.frequency_volume

        pnt.interpolate_volume(
            scale=scale,
            missing_fill=0.0,
            do_interpolation=False,
            non_linear=False,
            value_mode="object_count",
        )
        gv_obj, fv2 = pnt.interpolated_volume, pnt.frequency_volume

        with tempfile.TemporaryDirectory(prefix="pynutil_interpolate_value_modes_object_") as tmpdir:
            out_root = os.path.join(tmpdir, "object_count")

            # Save pixel_count
            pnt.interpolate_volume(
                scale=scale,
                missing_fill=0.0,
                do_interpolation=False,
                non_linear=False,
                value_mode="pixel_count",
            )
            pnt.save_analysis(os.path.join(out_root, "pixel_count"), create_visualisations=True)

            # Save object_count
            pnt.interpolate_volume(
                scale=scale,
                missing_fill=0.0,
                do_interpolation=False,
                non_linear=False,
                value_mode="object_count",
            )
            pnt.save_analysis(os.path.join(out_root, "object_count"), create_visualisations=True)

            copy_tree_to_demo(
                out_root,
                os.path.join(self.demo_outputs_root, "object_count"),
            )

        # NOTE: NumPy 2.2.6 on Python 3.14 has a regression where
        # np.testing.assert_array_equal can fail even for equal arrays.
        self.assertTrue(np.array_equal(fv, fv2))

        # Object counts should be non-negative integers.
        self.assertTrue(np.all(gv_obj >= 0.0))
        self.assertTrue(np.all(np.isclose(gv_obj, np.round(gv_obj))))

        # If there are no segmented pixels, there cannot be any objects.
        self.assertTrue(np.all(gv_obj[gv_pc == 0] == 0))

        # Each object contributes at least 1 segmented pixel.
        self.assertTrue(np.all(gv_obj <= gv_pc + 1e-6))
        # Each object contributes at least 1 pixel overall.
        self.assertTrue(np.all(gv_obj <= fv.astype(np.float32) + 1e-6))


if __name__ == "__main__":
    unittest.main()

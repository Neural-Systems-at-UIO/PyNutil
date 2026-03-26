import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

from PyNutil import read_alignment, seg_to_coords, quantify_coords, save_analysis, interpolate_volume
from tests.test_helpers import copy_tree_to_demo, small_volume_scale

try:
    # When run via `python -m unittest discover` from repo root
    from tests.timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    # When run with tests/ on sys.path
    from timing_utils import TimedTestCase


class TestInterpolateVolumeValueModes(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(os.path.dirname(__file__))
        self.settings_path = os.path.join(
            self.tests_dir, "test_cases", "brainglobe_atlas.json"
        )

        self.demo_outputs_root = os.path.join(
            self.tests_dir, "..", "demo_data", "outputs", "interpolate_value_modes"
        )

    def _load_settings(self):
        with open(self.settings_path) as f:
            return json.load(f)

    def _run_pipeline(self):
        settings = self._load_settings()
        atlas = BrainGlobeAtlas(settings["atlas_name"])
        alignment = read_alignment(settings["alignment_json"])
        result = seg_to_coords(
            settings["segmentation_folder"],
            alignment,
            atlas,
            pixel_id=settings.get("colour", [0, 0, 0]),
        )
        label_df = quantify_coords(result, atlas)
        return settings, atlas, result, label_df

    def _scale_for_small_volume(self, atlas_shape):
        return small_volume_scale(atlas_shape)

    def test_value_mode_mean_matches_pixel_count_over_frequency(self):
        settings, atlas, result, label_df = self._run_pipeline()

        scale = self._scale_for_small_volume(atlas.annotation.shape)

        common_kwargs = dict(
            segmentation_folder=settings["segmentation_folder"],
            alignment_json=settings["alignment_json"],
            colour=settings.get("colour", [0, 0, 0]),
            atlas=atlas,
            scale=scale,
            do_interpolation=False,
            non_linear=False,
            segmentation_format="binary",
            segmentation_mode=True,
        )

        gv_pc, fv, _ = interpolate_volume(
            **common_kwargs,
            missing_fill=0.0,
            value_mode="pixel_count",
        )

        gv_mean, fv2, _ = interpolate_volume(
            **common_kwargs,
            missing_fill=-1.0,
            value_mode="mean",
        )

        with tempfile.TemporaryDirectory(prefix="pynutil_interpolate_value_modes_mean_") as tmpdir:
            out_root = os.path.join(tmpdir, "mean_vs_pixel_count")

            # Save pixel_count
            save_analysis(
                os.path.join(out_root, "pixel_count"),
                result, atlas, label_df,
            )

            # Save mean
            save_analysis(
                os.path.join(out_root, "mean"),
                result, atlas, label_df,
            )

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
        settings, atlas, result, label_df = self._run_pipeline()

        scale = self._scale_for_small_volume(atlas.annotation.shape)

        common_kwargs = dict(
            segmentation_folder=settings["segmentation_folder"],
            alignment_json=settings["alignment_json"],
            colour=settings.get("colour", [0, 0, 0]),
            atlas=atlas,
            scale=scale,
            do_interpolation=False,
            non_linear=False,
            segmentation_format="binary",
            segmentation_mode=True,
            missing_fill=0.0,
        )

        gv_pc, fv, _ = interpolate_volume(
            **common_kwargs,
            value_mode="pixel_count",
        )

        gv_obj, fv2, _ = interpolate_volume(
            **common_kwargs,
            value_mode="object_count",
        )

        with tempfile.TemporaryDirectory(prefix="pynutil_interpolate_value_modes_object_") as tmpdir:
            out_root = os.path.join(tmpdir, "object_count")

            # Save pixel_count
            save_analysis(
                os.path.join(out_root, "pixel_count"),
                result, atlas, label_df,
            )

            # Save object_count
            save_analysis(
                os.path.join(out_root, "object_count"),
                result, atlas, label_df,
            )

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

    def test_colour_auto_matches_adapter_auto_detection(self):
        settings = self._load_settings()
        atlas = BrainGlobeAtlas(settings["atlas_name"])
        scale = self._scale_for_small_volume(atlas.annotation.shape)

        common_kwargs = dict(
            segmentation_folder=settings["segmentation_folder"],
            alignment_json=settings["alignment_json"],
            atlas=atlas,
            scale=scale,
            missing_fill=0.0,
            do_interpolation=False,
            non_linear=False,
            segmentation_format="binary",
            segmentation_mode=True,
            value_mode="pixel_count",
        )

        gv_auto, fv_auto, dv_auto = interpolate_volume(
            **common_kwargs,
            colour="auto",
        )
        gv_none, fv_none, dv_none = interpolate_volume(
            **common_kwargs,
            colour=None,
        )

        self.assertTrue(np.array_equal(gv_auto, gv_none))
        self.assertTrue(np.array_equal(fv_auto, fv_none))
        self.assertTrue(np.array_equal(dv_auto, dv_none))


if __name__ == "__main__":
    unittest.main()

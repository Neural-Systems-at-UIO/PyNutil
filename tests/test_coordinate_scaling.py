import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

from timing_utils import TimedTestCase

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyNutil.processing.coordinate_extraction import segmentation_to_atlas_space
from PyNutil.processing.transform import get_region_areas


class TestCoordinateScaling(TimedTestCase):
    def test_segmentation_size_mismatch_does_not_swap_axes(self):
        """Regression test: segmentation (h,w) may differ from alignment JSON (H,W).

        We should scale y by H/h and x by W/w. A previous bug swapped width/height
        when calling transform_to_registration, which mis-assigned pixels to regions
        when dimensions differed.
        """

        tmpdir = tempfile.mkdtemp(prefix="pynutil_test_")
        try:
            # Segmentation is (height=50, width=100)
            seg_h, seg_w = 50, 100
            segmentation = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
            # Use BGR since cv2.imread returns BGR
            segmentation[10, 25] = (0, 0, 255)
            seg_path = os.path.join(tmpdir, "seg.png")
            self.assertTrue(cv2.imwrite(seg_path, segmentation))

            # Alignment JSON (registration space) is (height=200, width=400)
            reg_h, reg_w = 200, 400
            slice_dict = {
                "width": reg_w,
                "height": reg_h,
                # Anchoring is not relevant for this test (we patch atlas generation).
                "anchoring": [0, 0, 0, reg_w, 0, 0, 0, reg_h, 0],
            }

            # Dummy atlas map: left half = 1, right half = 2
            atlas_map = np.ones((reg_h, reg_w), dtype=np.int32)
            atlas_map[:, reg_w // 2 :] = 2

            # Storage arrays expected by segmentation_to_atlas_space
            points_list = [None]
            centroids_list = [None]
            points_labels = [None]
            centroids_labels = [None]
            region_areas_list = [None]
            per_point_undamaged_list = [None]
            per_centroid_undamaged_list = [None]
            points_hemi_labels = [None]
            centroids_hemi_labels = [None]

            expected_y_scale = reg_h / seg_h
            expected_x_scale = reg_w / seg_w

            def _fake_get_region_areas(*args, **kwargs):
                return None, atlas_map

            def _fake_get_objects(
                segmentation_img,
                pixel_id,
                atlas_map_img,
                y_scale,
                x_scale,
                **kwargs,
            ):
                self.assertAlmostEqual(y_scale, expected_y_scale)
                self.assertAlmostEqual(x_scale, expected_x_scale)

                # Return one point/centroid with known scaled coords
                scaled_y = np.array([10 * expected_y_scale], dtype=float)
                scaled_x = np.array([25 * expected_x_scale], dtype=float)
                centroids = np.array([[10.0, 25.0]], dtype=float)
                scaled_centroidsX = scaled_x.copy()
                scaled_centroidsY = scaled_y.copy()
                per_centroid_labels = np.array([1], dtype=atlas_map.dtype)

                return (
                    centroids,
                    scaled_centroidsX,
                    scaled_centroidsY,
                    scaled_y,
                    scaled_x,
                    per_centroid_labels,
                )

            with (
                patch(
                    "PyNutil.processing.coordinate_extraction.get_region_areas",
                    side_effect=_fake_get_region_areas,
                ),
                patch(
                    "PyNutil.processing.coordinate_extraction.get_objects_and_assign_regions_optimized",
                    side_effect=_fake_get_objects,
                ),
            ):
                segmentation_to_atlas_space(
                    slice_dict=slice_dict,
                    segmentation_path=seg_path,
                    atlas_labels=pd.DataFrame(),
                    flat_file_atlas=None,
                    pixel_id="auto",
                    non_linear=False,
                    points_list=points_list,
                    centroids_list=centroids_list,
                    points_labels=points_labels,
                    centroids_labels=centroids_labels,
                    region_areas_list=region_areas_list,
                    per_point_undamaged_list=per_point_undamaged_list,
                    per_centroid_undamaged_list=per_centroid_undamaged_list,
                    points_hemi_labels=points_hemi_labels,
                    centroids_hemi_labels=centroids_hemi_labels,
                    index=0,
                    object_cutoff=0,
                    atlas_volume=None,
                    hemi_map=None,
                    use_flat=False,
                    grid_spacing=None,
                )

            # The single pixel at x=25 should land in the left half (label 1).
            self.assertIsNotNone(points_labels[0])
            self.assertEqual(int(points_labels[0][0]), 1)

        finally:
            shutil.rmtree(tmpdir)

    def test_warping_uses_registration_dims_not_segmentation_dims(self):
        """Regression test: atlas warping (non-linear) must use alignment JSON dims.

        If segmentation dims are passed into warping, region assignment can be wrong
        while atlas-space coordinates remain correct.
        """

        seg_w, seg_h = 100, 50
        reg_w, reg_h = 400, 200
        slice_dict = {
            "width": reg_w,
            "height": reg_h,
            "anchoring": [0, 0, 0, reg_w, 0, 0, 0, reg_h, 0],
        }

        atlas_labels = pd.DataFrame(
            {"idx": [0], "name": ["root"], "r": [0], "g": [0], "b": [0]}
        )

        called = {"warp_rescaleXY": None, "area_rescaleXY": None}

        def _fake_generate_target_slice(anchoring, atlas_volume):
            # Return a dummy slice in registration shape
            return np.zeros((reg_h, reg_w), dtype=np.uint32)

        def _fake_warp_image(image, triangulation, rescaleXY):
            called["warp_rescaleXY"] = rescaleXY
            return image

        def _fake_flat_to_dataframe(image, damage_mask, hemi_mask, rescaleXY=None):
            called["area_rescaleXY"] = rescaleXY
            return pd.DataFrame({"idx": [0], "region_area": [0]})

        with (
            patch(
                "PyNutil.processing.counting_and_load.generate_target_slice",
                side_effect=_fake_generate_target_slice,
            ),
            patch(
                "PyNutil.processing.counting_and_load.warp_image",
                side_effect=_fake_warp_image,
            ),
            patch(
                "PyNutil.processing.counting_and_load.flat_to_dataframe",
                side_effect=_fake_flat_to_dataframe,
            ),
        ):
            # Provide a non-None triangulation to force warp path
            get_region_areas(
                use_flat=False,
                atlas_labels=atlas_labels,
                flat_file_atlas=None,
                seg_width=seg_w,
                seg_height=seg_h,
                slice_dict=slice_dict,
                atlas_volume=np.zeros((2, 2, 2), dtype=np.uint32),
                hemi_mask=None,
                triangulation=object(),
                damage_mask=None,
            )

        self.assertEqual(called["warp_rescaleXY"], (reg_w, reg_h))
        self.assertEqual(called["area_rescaleXY"], (seg_w, seg_h))


if __name__ == "__main__":
    unittest.main()

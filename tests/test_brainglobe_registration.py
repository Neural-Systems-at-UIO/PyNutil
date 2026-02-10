import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
import cv2
import tifffile


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyNutil import PyNutil


def _load_registry_module():
    """Load adapters.registry without importing top-level PyNutil package."""
    pkg = types.ModuleType("PyNutil")
    pkg.__path__ = [str(REPO_ROOT / "PyNutil")]
    sys.modules.setdefault("PyNutil", pkg)

    processing_pkg = types.ModuleType("PyNutil.processing")
    processing_pkg.__path__ = [str(REPO_ROOT / "PyNutil" / "processing")]
    sys.modules.setdefault("PyNutil.processing", processing_pkg)

    adapters_pkg = types.ModuleType("PyNutil.processing.adapters")
    adapters_pkg.__path__ = [str(REPO_ROOT / "PyNutil" / "processing" / "adapters")]
    sys.modules.setdefault("PyNutil.processing.adapters", adapters_pkg)

    def _load(module_name: str, relative_path: str):
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(
            module_name, REPO_ROOT / relative_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    _load("PyNutil.processing.adapters.base", "PyNutil/processing/adapters/base.py")
    _load(
        "PyNutil.processing.adapters.visualign_deformations",
        "PyNutil/processing/adapters/visualign_deformations.py",
    )
    _load(
        "PyNutil.processing.adapters.deformation",
        "PyNutil/processing/adapters/deformation.py",
    )
    _load("PyNutil.processing.adapters.damage", "PyNutil/processing/adapters/damage.py")
    _load(
        "PyNutil.processing.adapters.anchoring",
        "PyNutil/processing/adapters/anchoring.py",
    )
    return _load(
        "PyNutil.processing.adapters.registry", "PyNutil/processing/adapters/registry.py"
    )


class TestBrainGlobeRegistration(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.join(
            os.path.dirname(__file__), "test_data", "brainglobe_registration_output"
        )
        self.registration_json = os.path.join(
            self.base_dir, "brainglobe-registration.json"
        )
        self.manual_output_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "demo_data",
            "outputs",
            "brainglobe_registration",
        )
        os.makedirs(self.manual_output_dir, exist_ok=True)
        self.registry = _load_registry_module()

    def test_loads_brainglobe_registration_as_single_slice(self):
        data = self.registry.load_registration(
            self.registration_json, apply_deformation=False
        )

        self.assertEqual(len(data.slices), 1)
        section = data.slices[0]

        self.assertEqual(section.width, 428)
        self.assertEqual(section.height, 318)

        expected_anchoring = np.array(
            [464.0, 207.0, 352.0, -475.0, 17.0, 2.0, 0.0, 63.0, -394.0],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            np.asarray(section.anchoring, dtype=np.float64),
            expected_anchoring,
            rtol=0,
            atol=1e-6,
        )

        with open(
            os.path.join(self.manual_output_dir, "anchoring_validation.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "registration_json": self.registration_json,
                    "width": int(section.width),
                    "height": int(section.height),
                    "computed_anchoring": [float(v) for v in section.anchoring],
                    "expected_anchoring": [float(v) for v in expected_anchoring],
                },
                f,
                indent=2,
            )

    def test_registered_intensity_is_mostly_within_registered_atlas_mask(self):
        image = tifffile.imread(os.path.join(self.base_dir, "downsampled.tiff"))
        hemispheres = tifffile.imread(
            os.path.join(self.base_dir, "registered_hemispheres.tiff")
        )

        foreground = image > 10
        inside_mask = hemispheres > 0

        coverage = float(np.mean(inside_mask[foreground]))
        self.assertGreaterEqual(coverage, 0.98)

        with open(
            os.path.join(self.manual_output_dir, "coverage_validation.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "foreground_threshold": 10,
                    "foreground_pixel_count": int(np.sum(foreground)),
                    "inside_mask_pixel_count": int(np.sum(inside_mask)),
                    "coverage": coverage,
                },
                f,
                indent=2,
            )

    def test_registered_atlas_mask_captures_brighter_signal_than_background(self):
        image = tifffile.imread(os.path.join(self.base_dir, "downsampled.tiff"))
        registered_atlas = tifffile.imread(
            os.path.join(self.base_dir, "registered_atlas.tiff")
        )

        inside_mask = registered_atlas > 0
        outside_mask = ~inside_mask

        inside_mean = float(np.mean(image[inside_mask]))
        outside_mean = float(np.mean(image[outside_mask]))

        self.assertGreaterEqual(inside_mean, outside_mean * 5.0)

        with open(
            os.path.join(self.manual_output_dir, "atlas_mask_validation.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "inside_mask_pixel_count": int(np.sum(inside_mask)),
                    "outside_mask_pixel_count": int(np.sum(outside_mask)),
                    "inside_mean": inside_mean,
                    "outside_mean": outside_mean,
                    "inside_outside_mean_ratio": (
                        inside_mean / outside_mean if outside_mean else None
                    ),
                },
                f,
                indent=2,
            )

    def test_generated_atlas_mask_overlaps_registered_atlas(self):
        from PyNutil.io.atlas_loader import load_atlas_data
        from PyNutil.processing.atlas_map import generate_target_slice

        data = self.registry.load_registration(
            self.registration_json, apply_deformation=False
        )
        section = data.slices[0]
        atlas_name = section.metadata.get("atlas", "allen_mouse_25um")
        atlas_volume, _, _ = load_atlas_data(atlas_name)

        atlas_map = generate_target_slice(section.anchoring, atlas_volume)
        registered_atlas = tifffile.imread(
            os.path.join(self.base_dir, "registered_atlas.tiff")
        )

        reg_h, reg_w = registered_atlas.shape
        atlas_resized = cv2.resize(
            atlas_map.astype(np.float32),
            (reg_w, reg_h),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_expected = registered_atlas > 0
        mask_actual = atlas_resized > 0

        intersection = int(np.logical_and(mask_actual, mask_expected).sum())
        union = int(np.logical_or(mask_actual, mask_expected).sum())
        iou = float(intersection / union) if union else 0.0

        self.assertGreaterEqual(iou, 0.65)

        with open(
            os.path.join(self.manual_output_dir, "atlas_overlap_validation.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "atlas_name": atlas_name,
                    "registered_shape": [int(reg_h), int(reg_w)],
                    "atlas_map_shape": [
                        int(atlas_map.shape[0]),
                        int(atlas_map.shape[1]),
                    ],
                    "intersection": intersection,
                    "union": union,
                    "iou": iou,
                },
                f,
                indent=2,
            )

    def test_nonlinear_deformation_fields_are_present_and_match_registration_shape(self):
        field0 = tifffile.imread(os.path.join(self.base_dir, "deformation_field_0.tiff"))
        field1 = tifffile.imread(os.path.join(self.base_dir, "deformation_field_1.tiff"))

        self.assertEqual(field0.shape, (318, 428))
        self.assertEqual(field1.shape, (318, 428))
        self.assertTrue(np.isfinite(field0).all())
        self.assertTrue(np.isfinite(field1).all())

    def test_brainglobe_deformation_field_is_loaded_for_points(self):
        data = self.registry.load_registration(self.registration_json, apply_deformation=True)
        section = data.slices[0]
        deformation_type = section.metadata.get("deformation_type")
        self.assertTrue(
            deformation_type
            in {"brainglobe_displacement", "brainglobe_displacement_pending"}
        )

    def test_brainglobe_deformation_field_warp_matches_registered_atlas(self):
        from PyNutil.io.atlas_loader import load_atlas_data
        from PyNutil.processing.atlas_map import generate_target_slice, warp_image

        cases = [
            ("brainglobe_registration_output", "brainglobe_registration"),
            ("brainglobe_registration_no_nonlinear_output", "brainglobe_registration_no_nonlinear"),
        ]

        for folder_name, output_tag in cases:
            with self.subTest(case=folder_name):
                base_dir = os.path.join(os.path.dirname(__file__), "test_data", folder_name)
                reg_json = os.path.join(base_dir, "brainglobe-registration.json")
                data = self.registry.load_registration(reg_json, apply_deformation=False)
                section = data.slices[0]
                atlas_name = section.metadata.get("atlas", "allen_mouse_25um")
                atlas_volume, _, _ = load_atlas_data(atlas_name)

                atlas_map = generate_target_slice(section.anchoring, atlas_volume).astype(
                    np.float32
                )
                registered_atlas = tifffile.imread(
                    os.path.join(base_dir, "registered_atlas.tiff")
                )
                reg_h, reg_w = registered_atlas.shape

                field0 = tifffile.imread(os.path.join(base_dir, "deformation_field_0.tiff"))
                field1 = tifffile.imread(os.path.join(base_dir, "deformation_field_1.tiff"))
                field = np.stack([field0, field1], axis=-1)

                def _iou(mask_a, mask_b):
                    intersection = np.logical_and(mask_a, mask_b).sum()
                    union = np.logical_or(mask_a, mask_b).sum()
                    return float(intersection / union) if union else 0.0

                def _warp_iou(mode: str) -> float:
                    if mode == "xy":
                        dx = field[..., 1]
                        dy = field[..., 0]
                        sign = 1.0
                    elif mode == "xy_inv":
                        dx = field[..., 1]
                        dy = field[..., 0]
                        sign = -1.0
                    elif mode == "yx":
                        dx = field[..., 0]
                        dy = field[..., 1]
                        sign = 1.0
                    else:
                        dx = field[..., 0]
                        dy = field[..., 1]
                        sign = -1.0

                    def deform(x, y):
                        h, w = field.shape[:2]
                        xi = np.clip(x.astype(np.int32), 0, w - 1)
                        yi = np.clip(y.astype(np.int32), 0, h - 1)
                        return x + sign * dx[yi, xi], y + sign * dy[yi, xi]

                    warped = warp_image(atlas_map, deform, (reg_w, reg_h))
                    if warped.shape != (reg_h, reg_w):
                        warped = cv2.resize(
                            warped.astype(np.float32),
                            (reg_w, reg_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    return _iou(warped > 0, registered_atlas > 0)

                modes = ["xy", "xy_inv", "yx", "yx_inv"]
                scores = {mode: _warp_iou(mode) for mode in modes}
                best_mode = max(scores, key=scores.get)
                best_iou = scores[best_mode]

                with open(
                    os.path.join(
                        self.manual_output_dir,
                        f"deformation_warp_validation_{output_tag}.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "case": folder_name,
                            "best_mode": best_mode,
                            "best_iou": best_iou,
                            "mode_scores": scores,
                        },
                        f,
                        indent=2,
                    )

                self.assertGreaterEqual(best_iou, 0.85)

    def test_intensity_analysis_outputs_match_expected(self):
        with open(self.registration_json, "r", encoding="utf-8") as f:
            registration = json.load(f)

        atlas_name = registration.get("atlas", "allen_mouse_25um")
        section_number = int(registration.get("atlas_2d_slice_index", 0))

        tmp_image_dir = tempfile.mkdtemp(prefix="pynutil_bg_intensity_")
        try:
            src_image = os.path.join(self.base_dir, "downsampled.tiff")
            image_name = f"downsampled_s{section_number:03d}.tiff"
            dst_image = os.path.join(tmp_image_dir, image_name)
            shutil.copy(src_image, dst_image)

            output_root = os.path.join(
                os.path.dirname(__file__),
                "..",
                "demo_data",
                "outputs",
                "brainglobe_registration_intensity",
            )
            if os.path.exists(output_root):
                shutil.rmtree(output_root)

            pnt = PyNutil(
                image_folder=tmp_image_dir,
                intensity_channel="grayscale",
                alignment_json=self.registration_json,
                atlas_name=atlas_name,
            )
            pnt.get_coordinates()
            pnt.quantify_coordinates()
            pnt.save_analysis(output_root, create_visualisations=False)

            expected_root = os.path.join(
                os.path.dirname(__file__),
                "expected_outputs",
                "brainglobe_registration_intensity",
            )
            expected_files = [
                os.path.join(expected_root, "whole_series_report", "intensity.csv"),
                os.path.join(
                    expected_root,
                    "whole_series_meshview",
                    "pixels_meshview.json",
                ),
                os.path.join(
                    expected_root,
                    "whole_series_meshview",
                    "left_hemisphere_pixels_meshview.json",
                ),
                os.path.join(
                    expected_root,
                    "whole_series_meshview",
                    "right_hemisphere_pixels_meshview.json",
                ),
            ]
            actual_files = [
                os.path.join(output_root, "whole_series_report", "intensity.csv"),
                os.path.join(
                    output_root, "whole_series_meshview", "pixels_meshview.json"
                ),
                os.path.join(
                    output_root,
                    "whole_series_meshview",
                    "left_hemisphere_pixels_meshview.json",
                ),
                os.path.join(
                    output_root,
                    "whole_series_meshview",
                    "right_hemisphere_pixels_meshview.json",
                ),
            ]

            for expected_path, actual_path in zip(expected_files, actual_files):
                with self.subTest(expected=expected_path):
                    self.assertTrue(
                        os.path.exists(expected_path),
                        f"Expected output missing: {expected_path}",
                    )
                    self.assertTrue(
                        os.path.exists(actual_path),
                        f"Actual output missing: {actual_path}",
                    )
                    with open(expected_path, "rb") as f_exp, open(
                        actual_path, "rb"
                    ) as f_act:
                        self.assertEqual(
                            f_exp.read(),
                            f_act.read(),
                            f"Output differs for {os.path.basename(expected_path)}",
                        )
        finally:
            shutil.rmtree(tmp_image_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest

import cv2
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

from PyNutil import read_alignment, read_segmentation_dir, seg_to_coords
from PyNutil.io.atlas_loader import resolve_atlas
from timing_utils import TimedTestCase


class TestVisualisations(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(os.path.dirname(__file__))
        self.expected_dir = os.path.join(
            self.tests_dir, "test_data", "nonlinear_allen_mouse", "visualisations"
        )
        self.settings_path = os.path.join(
            self.tests_dir, "test_cases", "brainglobe_atlas.json"
        )

    def _load_rgb(self, path: str) -> np.ndarray:
        im_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    def test_generated_visualisations_match_expected(self):
        if not os.path.isdir(self.expected_dir):
            self.skipTest(
                f"Expected visualisations folder not found: {self.expected_dir}"
            )

        expected_files = sorted(
            f for f in os.listdir(self.expected_dir) if f.endswith("_atlas_colored.png")
        )
        if not expected_files:
            self.skipTest("No expected *_atlas_colored.png files found")

        # Generate visualisations into a temp folder
        with tempfile.TemporaryDirectory() as tmp:
            output_root = os.path.join(tmp, "outputs")

            with open(self.settings_path) as f:
                settings = json.load(f)

            atlas = resolve_atlas(BrainGlobeAtlas(settings["atlas_name"]))
            alignment = read_alignment(settings["alignment_json"])
            result = seg_to_coords(
                read_segmentation_dir(
                    settings["segmentation_folder"],
                    pixel_id=settings.get("colour", [0, 0, 0]),
                    segmentation_format=settings.get("segmentation_format", "binary"),
                ),
                alignment,
                atlas,
            )

            from PyNutil.processing.adapters.segmentation import SegmentationAdapterRegistry
            from PyNutil.io.section_visualisation import create_section_visualisations

            reg_data = read_alignment(
                settings["alignment_json"], apply_deformation=False, apply_damage=False
            )
            adapter = SegmentationAdapterRegistry.get(settings.get("segmentation_format", "binary"))

            create_section_visualisations(
                settings["segmentation_folder"],
                reg_data.slices,
                atlas.volume,
                atlas.labels,
                output_root,
                adapter=adapter,
                pixel_id=settings.get("colour", [0, 0, 0]),
            )

            generated_dir = os.path.join(output_root, "visualisations")
            self.assertTrue(
                os.path.isdir(generated_dir),
                f"Generated visualisations folder missing: {generated_dir}",
            )

            # Compare each expected file against the generated one
            for filename in expected_files:
                expected_path = os.path.join(self.expected_dir, filename)
                generated_path = os.path.join(generated_dir, filename)

                self.assertTrue(
                    os.path.exists(generated_path),
                    f"Expected generated file missing: {generated_path}",
                )

                exp = self._load_rgb(expected_path)
                got = self._load_rgb(generated_path)

                self.assertEqual(
                    exp.shape,
                    got.shape,
                    f"Image shape mismatch for {filename}: expected {exp.shape}, got {got.shape}",
                )

                if not np.array_equal(exp, got):
                    diff = np.abs(exp.astype(np.int16) - got.astype(np.int16))
                    max_diff = int(diff.max())
                    mean_diff = float(diff.mean())
                    self.fail(
                        f"Image pixels differ for {filename}: max_diff={max_diff}, mean_diff={mean_diff:.4f}"
                    )


if __name__ == "__main__":
    unittest.main()

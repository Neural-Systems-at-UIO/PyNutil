import os
import tempfile
import unittest

import cv2
import numpy as np

from PyNutil import PyNutil
from timing_utils import TimedTestCase


class TestVisualisations(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(__file__)
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

            pnt = PyNutil(settings_file=self.settings_path)

            from PyNutil.io.read_and_write import load_quint_json
            from PyNutil.io.section_visualisation import create_section_visualisations

            alignment_data = load_quint_json(pnt.alignment_json)

            create_section_visualisations(
                pnt.segmentation_folder,
                alignment_data,
                pnt.atlas_volume,
                pnt.atlas_labels,
                output_root,
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

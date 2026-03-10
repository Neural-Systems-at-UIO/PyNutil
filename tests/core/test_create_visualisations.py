"""Tests for _create_visualisations via the adapter system.

Validates that visualisation generation works for all registration formats
by going through load_registration (the generic adapter pipeline) rather
than format-specific loaders.

Scenarios covered:
1. QuickNII/VisuAlign alignment (segmentation workflow)
2. Brainglobe registration (segmentation workflow)
3. Brainglobe registration (coordinate workflow, no segmentation folder)
4. Direct create_section_visualisations with None segmentation folder
5. _create_visualisations produces valid PNG output
"""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd


TEST_DIR = os.path.dirname(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)


class TestCreateVisualisationsAdapter(unittest.TestCase):
    """Test that _create_visualisations works for all registration formats."""

    def test_quint_alignment_produces_visualisations(self):
        """QuickNII/VisuAlign alignment should produce visualisation PNGs
        by calling _create_visualisations directly."""
        from PyNutil.io.section_visualisation import create_section_visualisations
        from PyNutil.processing.adapters.registry import load_registration
        from PyNutil.io.atlas_loader import load_atlas_data

        alignment_json = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "alignment.json"
        )
        seg_folder = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "segmentations"
        )
        if not os.path.isfile(alignment_json):
            self.skipTest("QuickNII alignment test data not found")

        atlas_volume, _, atlas_labels = load_atlas_data("allen_mouse_25um")
        reg_data = load_registration(
            alignment_json, apply_deformation=False, apply_damage=False
        )
        alignment_data = {
            "slices": [
                {
                    "filename": s.section_id,
                    "nr": s.section_number,
                    "anchoring": s.anchoring,
                    "width": s.width,
                    "height": s.height,
                }
                for s in reg_data.slices
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                seg_folder, alignment_data, atlas_volume, atlas_labels, tmp,
            )
            viz_dir = os.path.join(tmp, "visualisations")
            self.assertTrue(os.path.isdir(viz_dir))
            pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
            self.assertGreater(len(pngs), 0, "No visualisation PNGs generated")
            for png in pngs:
                img = cv2.imread(os.path.join(viz_dir, png))
                self.assertIsNotNone(img, f"Failed to read {png}")
                self.assertGreater(img.size, 0)

    def test_brainglobe_registration_produces_visualisations(self):
        """Brainglobe registration should produce visualisation PNGs
        (without segmentation overlay, since no matching segmentations exist)."""
        from PyNutil.io.section_visualisation import create_section_visualisations
        from PyNutil.processing.adapters.registry import load_registration
        from PyNutil.io.atlas_loader import load_atlas_data

        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_registration", "brainglobe-registration.json"
        )
        if not os.path.isfile(bg_json):
            self.skipTest("Brainglobe registration test data not found")

        atlas_volume, _, atlas_labels = load_atlas_data("allen_mouse_25um")
        reg_data = load_registration(
            bg_json, apply_deformation=False, apply_damage=False
        )
        alignment_data = {
            "slices": [
                {
                    "filename": s.section_id,
                    "nr": s.section_number,
                    "anchoring": s.anchoring,
                    "width": s.width,
                    "height": s.height,
                }
                for s in reg_data.slices
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                None, alignment_data, atlas_volume, atlas_labels, tmp,
            )
            viz_dir = os.path.join(tmp, "visualisations")
            self.assertTrue(os.path.isdir(viz_dir))
            pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
            self.assertGreater(len(pngs), 0, "No visualisation PNGs generated")
            for png in pngs:
                img = cv2.imread(os.path.join(viz_dir, png))
                self.assertIsNotNone(img, f"Failed to read {png}")
                self.assertGreater(np.count_nonzero(img), 0, f"{png} is blank")

    def test_brainglobe_coordinate_workflow_produces_visualisations(self):
        """Brainglobe registration with coordinates (no segmentation folder)
        should produce visualisation PNGs without errors."""
        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_coordinates", "brainglobe-registration.json"
        )
        coord_file = os.path.join(
            TEST_DIR, "test_data", "brainglobe_coordinates", "coordinates.csv"
        )
        if not os.path.isfile(bg_json) or not os.path.isfile(coord_file):
            self.skipTest("Brainglobe coordinate test data not found")

        from PyNutil import PyNutil

        pnt = PyNutil(
            coordinate_file=coord_file,
            alignment_json=bg_json,
            atlas_name="allen_mouse_25um",
        )
        pnt.get_coordinates()
        pnt.quantify_coordinates()

        with tempfile.TemporaryDirectory() as tmp:
            pnt.save_analysis(tmp, create_visualisations=True)
            viz_dir = os.path.join(tmp, "visualisations")
            self.assertTrue(os.path.isdir(viz_dir))
            pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
            self.assertGreater(len(pngs), 0, "No visualisation PNGs generated")
            # Verify the image has actual content (not blank)
            for png in pngs:
                img = cv2.imread(os.path.join(viz_dir, png))
                self.assertIsNotNone(img)
                self.assertGreater(
                    np.count_nonzero(img), 0, f"{png} is blank"
                )


class TestCreateSectionVisualisationsNoneFolder(unittest.TestCase):
    """Test create_section_visualisations handles None segmentation folder."""

    def test_none_segmentation_folder(self):
        """Passing None as segmentation_folder should not raise."""
        from PyNutil.io.section_visualisation import create_section_visualisations
        from PyNutil.io.atlas_loader import load_atlas_data

        atlas_volume, _, atlas_labels = load_atlas_data("allen_mouse_25um")

        # Minimal slice data with a valid anchoring vector
        alignment_data = {
            "slices": [
                {
                    "filename": "test_slice",
                    "nr": 0,
                    "anchoring": [
                        200, 200, 200,
                        0, 0, 100,
                        0, 100, 0,
                    ],
                    "width": 100,
                    "height": 100,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                None,
                alignment_data,
                atlas_volume,
                atlas_labels,
                tmp,
            )
            viz_dir = os.path.join(tmp, "visualisations")
            self.assertTrue(os.path.isdir(viz_dir))
            pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
            self.assertEqual(len(pngs), 1)

    def test_empty_slices_list(self):
        """An empty slices list should produce no output and not raise."""
        from PyNutil.io.section_visualisation import create_section_visualisations
        from PyNutil.io.atlas_loader import load_atlas_data

        atlas_volume, _, atlas_labels = load_atlas_data("allen_mouse_25um")

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                None,
                {"slices": []},
                atlas_volume,
                atlas_labels,
                tmp,
            )
            viz_dir = os.path.join(tmp, "visualisations")
            self.assertTrue(os.path.isdir(viz_dir))
            pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
            self.assertEqual(len(pngs), 0)


class TestLoadRegistrationForVisualisation(unittest.TestCase):
    """Test that load_registration returns data usable for visualisation."""

    def _load_and_convert(self, alignment_json):
        """Load registration and convert to visualisation dict format."""
        from PyNutil.processing.adapters.registry import load_registration

        reg_data = load_registration(
            alignment_json, apply_deformation=False, apply_damage=False
        )
        return {
            "slices": [
                {
                    "filename": s.section_id,
                    "nr": s.section_number,
                    "anchoring": s.anchoring,
                    "width": s.width,
                    "height": s.height,
                }
                for s in reg_data.slices
            ]
        }

    def test_quint_json_produces_valid_slices(self):
        """QuickNII JSON should produce slices with all required fields."""
        quint_json = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "alignment.json"
        )
        if not os.path.isfile(quint_json):
            self.skipTest("QuickNII alignment test data not found")

        result = self._load_and_convert(quint_json)
        self.assertIn("slices", result)
        self.assertGreater(len(result["slices"]), 0)
        for s in result["slices"]:
            self.assertIn("anchoring", s)
            self.assertEqual(len(s["anchoring"]), 9)
            self.assertIn("width", s)
            self.assertIn("height", s)
            self.assertGreater(s["width"], 0)
            self.assertGreater(s["height"], 0)

    def test_brainglobe_json_produces_valid_slices(self):
        """Brainglobe registration JSON should produce slices with all required fields."""
        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_registration", "brainglobe-registration.json"
        )
        if not os.path.isfile(bg_json):
            self.skipTest("Brainglobe registration test data not found")

        result = self._load_and_convert(bg_json)
        self.assertIn("slices", result)
        self.assertGreater(len(result["slices"]), 0)
        for s in result["slices"]:
            self.assertIn("anchoring", s)
            self.assertEqual(len(s["anchoring"]), 9)
            self.assertIn("width", s)
            self.assertIn("height", s)
            self.assertGreater(s["width"], 0)
            self.assertGreater(s["height"], 0)

    def test_brainglobe_coordinate_json_produces_valid_slices(self):
        """Brainglobe coordinate registration JSON should also work."""
        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_coordinates", "brainglobe-registration.json"
        )
        if not os.path.isfile(bg_json):
            self.skipTest("Brainglobe coordinate test data not found")

        result = self._load_and_convert(bg_json)
        self.assertIn("slices", result)
        self.assertGreater(len(result["slices"]), 0)
        for s in result["slices"]:
            self.assertEqual(len(s["anchoring"]), 9)
            self.assertGreater(s["width"], 0)
            self.assertGreater(s["height"], 0)


if __name__ == "__main__":
    unittest.main()

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
from brainglobe_atlasapi import BrainGlobeAtlas


TEST_DIR = os.path.dirname(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)


class TestCreateVisualisationsAdapter(unittest.TestCase):
    """Test that _create_visualisations works for all registration formats."""

    def test_quint_alignment_produces_visualisations(self):
        """QuickNII/VisuAlign alignment should produce visualisation PNGs
        by calling _create_visualisations directly."""
        from PyNutil.io.section_visualisation import create_section_visualisations
        from PyNutil.processing.adapters.registry import read_alignment
        from PyNutil.io.atlas_loader import resolve_atlas

        alignment_json = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "alignment.json"
        )
        seg_folder = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "segmentations"
        )
        if not os.path.isfile(alignment_json):
            self.skipTest("QuickNII alignment test data not found")

        _atlas = resolve_atlas(BrainGlobeAtlas("allen_mouse_25um"))
        atlas_volume, atlas_labels = _atlas.volume, _atlas.labels
        reg_data = read_alignment(
            alignment_json, apply_deformation=False, apply_damage=False
        )

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                seg_folder, reg_data.slices, atlas_volume, atlas_labels, tmp,
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
        from PyNutil.processing.adapters.registry import read_alignment
        from PyNutil.io.atlas_loader import resolve_atlas

        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_registration", "brainglobe-registration.json"
        )
        if not os.path.isfile(bg_json):
            self.skipTest("Brainglobe registration test data not found")

        _atlas = resolve_atlas(BrainGlobeAtlas("allen_mouse_25um"))
        atlas_volume, atlas_labels = _atlas.volume, _atlas.labels
        reg_data = read_alignment(
            bg_json, apply_deformation=False, apply_damage=False
        )

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                None, reg_data.slices, atlas_volume, atlas_labels, tmp,
            )
            viz_dir = os.path.join(tmp, "visualisations")
            self.assertTrue(os.path.isdir(viz_dir))
            pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
            self.assertGreater(len(pngs), 0, "No visualisation PNGs generated")
            for png in pngs:
                img = cv2.imread(os.path.join(viz_dir, png))
                self.assertIsNotNone(img, f"Failed to read {png}")
                self.assertGreater(np.count_nonzero(img), 0, f"{png} is blank")

    def test_brainglobe_coordinate_workflow_quantifies(self):
        """Brainglobe registration with coordinates produces quantification."""
        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_coordinates", "registration", "brainglobe-registration.json"
        )
        coord_file = os.path.join(
            TEST_DIR, "test_data", "brainglobe_coordinates", "coordinates.csv"
        )
        if not os.path.isfile(bg_json) or not os.path.isfile(coord_file):
            self.skipTest("Brainglobe coordinate test data not found")

        from PyNutil import read_alignment, xy_to_coords, quantify_coords, save_analysis

        atlas = BrainGlobeAtlas("allen_mouse_25um")
        alignment = read_alignment(bg_json)
        result = xy_to_coords(coord_file, alignment, atlas)
        label_df = quantify_coords(result, atlas)

        self.assertFalse(label_df.empty)

        with tempfile.TemporaryDirectory() as tmp:
            save_analysis(tmp, result, atlas, label_df)
            self.assertTrue(
                os.path.isfile(os.path.join(tmp, "whole_series_report", "counts.csv"))
                )


class TestCreateSectionVisualisationsNoneFolder(unittest.TestCase):
    """Test create_section_visualisations handles None segmentation folder."""

    def test_none_segmentation_folder(self):
        """Passing None as segmentation_folder should not raise."""
        from PyNutil.io.section_visualisation import create_section_visualisations
        from PyNutil.io.atlas_loader import resolve_atlas
        from PyNutil.processing.adapters.base import SliceInfo

        _atlas = resolve_atlas(BrainGlobeAtlas("allen_mouse_25um"))
        atlas_volume, atlas_labels = _atlas.volume, _atlas.labels

        slices = [
            SliceInfo(
                section_id="test_slice",
                section_number=0,
                width=100,
                height=100,
                anchoring=[200, 200, 200, 0, 0, 100, 0, 100, 0],
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                None,
                slices,
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
        from PyNutil.io.atlas_loader import resolve_atlas

        _atlas = resolve_atlas(BrainGlobeAtlas("allen_mouse_25um"))
        atlas_volume, atlas_labels = _atlas.volume, _atlas.labels

        with tempfile.TemporaryDirectory() as tmp:
            create_section_visualisations(
                None,
                [],
                atlas_volume,
                atlas_labels,
                tmp,
            )
            viz_dir = os.path.join(tmp, "visualisations")
            self.assertTrue(os.path.isdir(viz_dir))
            pngs = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
            self.assertEqual(len(pngs), 0)


class TestLoadRegistrationForVisualisation(unittest.TestCase):
    """Test that load_registration returns SliceInfo objects with the correct attributes."""

    def _load_slices(self, alignment_json):
        from PyNutil.processing.adapters.registry import read_alignment

        reg_data = read_alignment(
            alignment_json, apply_deformation=False, apply_damage=False
        )
        return reg_data.slices

    def test_quint_json_produces_valid_slices(self):
        """QuickNII JSON should produce SliceInfo objects with all required fields."""
        quint_json = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "alignment.json"
        )
        if not os.path.isfile(quint_json):
            self.skipTest("QuickNII alignment test data not found")

        slices = self._load_slices(quint_json)
        self.assertGreater(len(slices), 0)
        for s in slices:
            self.assertEqual(len(s.anchoring), 9)
            self.assertGreater(s.width, 0)
            self.assertGreater(s.height, 0)

    def test_brainglobe_json_produces_valid_slices(self):
        """Brainglobe registration JSON should produce SliceInfo with all required fields."""
        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_registration", "brainglobe-registration.json"
        )
        if not os.path.isfile(bg_json):
            self.skipTest("Brainglobe registration test data not found")

        slices = self._load_slices(bg_json)
        self.assertGreater(len(slices), 0)
        for s in slices:
            self.assertEqual(len(s.anchoring), 9)
            self.assertGreater(s.width, 0)
            self.assertGreater(s.height, 0)

    def test_brainglobe_coordinate_json_produces_valid_slices(self):
        """Brainglobe coordinate registration JSON should also work."""
        bg_json = os.path.join(
            TEST_DIR, "test_data", "brainglobe_coordinates", "registration", "brainglobe-registration.json"
        )
        if not os.path.isfile(bg_json):
            self.skipTest("Brainglobe coordinate test data not found")

        slices = self._load_slices(bg_json)
        self.assertGreater(len(slices), 0)
        for s in slices:
            self.assertEqual(len(s.anchoring), 9)
            self.assertGreater(s.width, 0)
            self.assertGreater(s.height, 0)


if __name__ == "__main__":
    unittest.main()

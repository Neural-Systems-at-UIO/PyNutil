import unittest
import os
import sys
import shutil
import json
import pandas as pd

# Add the root directory to sys.path to allow importing PyNutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyNutil import PyNutil
from tests.timing_utils import TimedTestCase

class TestIntensityQuantification(TimedTestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data", "image_intensity"))
        self.image_folder = os.path.join(self.base_dir, "images")
        self.rgb_image_folder = os.path.join(self.base_dir, "rgb_images")
        self.alignment_json = os.path.join(self.base_dir, "alignment.json")
        self.output_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_data", "outputs"))
        self.atlas_name = "allen_mouse_25um"

        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

    def test_intensity_quantification_magma(self):
        output_folder = os.path.join(self.output_root, "test_intensity_grayscale_magma")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Initialize PyNutil with a filter (e.g., min_intensity=20)
        pynutil = PyNutil(
            image_folder=self.image_folder,
            intensity_channel="grayscale",
            alignment_json=self.alignment_json,
            atlas_name=self.atlas_name,
            min_intensity=20
        )

        # Run quantification
        pynutil.get_coordinates()
        pynutil.quantify_coordinates()

        # Save with magma colormap
        pynutil.save_analysis(output_folder, colormap="magma")

        # Assertions
        self.assertTrue(os.path.exists(output_folder))
        meshview_json = os.path.join(output_folder, "whole_series_meshview", "pixels_meshview.json")
        self.assertTrue(os.path.exists(meshview_json))

        with open(meshview_json, "r") as f:
            data = json.load(f)
            self.assertGreater(len(data), 0)
            # Verify that no intensities below 20 are present
            for entry in data:
                # idx 0 is allowed as it's the background/filtered value
                self.assertTrue(entry["idx"] >= 20 or entry["idx"] == 0)

            high_intensity_entry = next((entry for entry in data if "Intensity 200" in entry["name"]), None)
            if high_intensity_entry:
                self.assertGreater(high_intensity_entry["r"], high_intensity_entry["g"])
                self.assertGreater(high_intensity_entry["r"], high_intensity_entry["b"])

    def test_intensity_quantification_rgb_original(self):
        output_folder = os.path.join(self.output_root, "test_intensity_rgb_original")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Initialize PyNutil
        pynutil = PyNutil(
            image_folder=self.rgb_image_folder,
            intensity_channel="grayscale",
            alignment_json=self.alignment_json,
            atlas_name=self.atlas_name
        )

        # Run quantification
        pynutil.get_coordinates()
        pynutil.quantify_coordinates()

        # Save with original_colours colormap
        pynutil.save_analysis(output_folder, colormap="original_colours")

        # Assertions
        self.assertTrue(os.path.exists(output_folder))
        meshview_json = os.path.join(output_folder, "whole_series_meshview", "pixels_meshview.json")
        self.assertTrue(os.path.exists(meshview_json))

        with open(meshview_json, "r") as f:
            data = json.load(f)
            self.assertGreater(len(data), 0)
            # Check for some RGB variety (not just grayscale)
            has_color = False
            for entry in data:
                if entry["idx"] > 0:
                    if not (entry["r"] == entry["g"] == entry["b"]):
                        has_color = True
                        break
            self.assertTrue(has_color, "RGB images should have some non-grayscale pixels in MeshView")

    def test_intensity_quantification_max_filter(self):
        output_folder = os.path.join(self.output_root, "test_intensity_max_filter")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Initialize PyNutil with a max filter (e.g., max_intensity=100)
        pynutil = PyNutil(
            image_folder=self.image_folder,
            intensity_channel="grayscale",
            alignment_json=self.alignment_json,
            atlas_name=self.atlas_name,
            max_intensity=100
        )

        # Run quantification
        pynutil.get_coordinates()
        pynutil.quantify_coordinates()
        pynutil.save_analysis(output_folder)

        # Assertions
        meshview_json = os.path.join(output_folder, "whole_series_meshview", "pixels_meshview.json")
        with open(meshview_json, "r") as f:
            data = json.load(f)
            self.assertGreater(len(data), 0)
            # Verify that no intensities above 100 are present
            for entry in data:
                self.assertLessEqual(entry["idx"], 100)

if __name__ == "__main__":
    unittest.main()

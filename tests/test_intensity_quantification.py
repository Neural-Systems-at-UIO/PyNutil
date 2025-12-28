import os
import pandas as pd
import json
import shutil
import sys

# Add the root directory to sys.path to allow importing PyNutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyNutil.main import PyNutil

def test_intensity_quantification_magma():
    # Setup paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data", "image_intensity"))
    image_folder = os.path.join(base_dir, "images")
    alignment_json = os.path.join(base_dir, "registration_data_combined_registration_jsons_05-2788.json")
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_data", "outputs", "test_intensity_grayscale_magma"))
    atlas_name = "allen_mouse_25um"

    # Clean up previous test output if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Initialize PyNutil
    pynutil = PyNutil(
        image_folder=image_folder,
        intensity_channel="grayscale",
        alignment_json=alignment_json,
        atlas_name=atlas_name
    )

    # Run quantification
    pynutil.get_coordinates()
    pynutil.quantify_coordinates()

    # Save with magma colormap
    pynutil.save_analysis(output_folder, colormap="magma")

    # Assertions for magma
    assert os.path.exists(output_folder)
    meshview_json = os.path.join(output_folder, "whole_series_meshview", "pixels_meshview.json")
    assert os.path.exists(meshview_json)
    
    # Check against expected output
    expected_csv = os.path.join(os.path.dirname(__file__), "expected_outputs", "intensity_quantification", "magma_quantification.csv")
    actual_csv = os.path.join(output_folder, "whole_series_report", "intensity.csv")
    expected_df = pd.read_csv(expected_csv, sep=";")
    actual_df = pd.read_csv(actual_csv, sep=";")
    pd.testing.assert_frame_equal(expected_df, actual_df)

    with open(meshview_json, "r") as f:
        data = json.load(f)
        assert len(data) > 0
        high_intensity_entry = next((entry for entry in data if "Intensity 200" in entry["name"]), None)
        if high_intensity_entry:
            assert high_intensity_entry["r"] > high_intensity_entry["g"]
            assert high_intensity_entry["r"] > high_intensity_entry["b"]

    # Test original_colours (should fall back to gray for these grayscale images)
    output_folder_orig = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_data", "outputs", "test_intensity_grayscale_original"))
    if os.path.exists(output_folder_orig):
        shutil.rmtree(output_folder_orig)
    pynutil.save_analysis(output_folder_orig, colormap="original_colours")

    meshview_json_orig = os.path.join(output_folder_orig, "whole_series_meshview", "pixels_meshview.json")
    assert os.path.exists(meshview_json_orig)
    with open(meshview_json_orig, "r") as f:
        data_orig = json.load(f)
        assert len(data_orig) > 0
        # For grayscale, r=g=b
        first_entry = data_orig[0]
        assert first_entry["r"] == first_entry["g"] == first_entry["b"]

def test_intensity_quantification_rgb_original():
    # Setup paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data", "image_intensity"))
    image_folder = os.path.join(base_dir, "rgb_images")
    alignment_json = os.path.join(base_dir, "registration_data_combined_registration_jsons_05-2788.json")
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_data", "outputs", "test_intensity_rgb_original"))
    atlas_name = "allen_mouse_25um"

    # Clean up previous test output if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Initialize PyNutil
    pynutil = PyNutil(
        image_folder=image_folder,
        intensity_channel="grayscale",
        alignment_json=alignment_json,
        atlas_name=atlas_name
    )

    # Run quantification
    pynutil.get_coordinates()
    pynutil.quantify_coordinates()

    # Save with original_colours colormap
    pynutil.save_analysis(output_folder, colormap="original_colours")

    # Assertions
    assert os.path.exists(output_folder)
    meshview_json = os.path.join(output_folder, "whole_series_meshview", "pixels_meshview.json")
    assert os.path.exists(meshview_json)
    
    # Check against expected output
    expected_csv = os.path.join(os.path.dirname(__file__), "expected_outputs", "intensity_quantification", "rgb_quantification.csv")
    actual_csv = os.path.join(output_folder, "whole_series_report", "intensity.csv")
    expected_df = pd.read_csv(expected_csv, sep=";")
    actual_df = pd.read_csv(actual_csv, sep=";")
    pd.testing.assert_frame_equal(expected_df, actual_df)

    with open(meshview_json, "r") as f:
        data = json.load(f)
        assert len(data) > 0
        # For RGB images, we expect some entries to have r != g or g != b
        has_color = False
        for entry in data:
            if not (entry["r"] == entry["g"] == entry["b"]):
                has_color = True
                break
        assert has_color, "Expected some non-grayscale colors in RGB MeshView output"

if __name__ == "__main__":
    test_intensity_quantification_magma()
    test_intensity_quantification_rgb_original()

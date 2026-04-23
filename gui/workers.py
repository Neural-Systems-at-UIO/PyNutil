import sys
from typing import Any, Dict

from brainglobe_atlasapi import BrainGlobeAtlas
from PyQt6.QtCore import QThread, pyqtSignal
from log_manager import TextRedirector
from PyNutil import (
    load_custom_atlas,
    resolve_atlas,
    read_alignment,
    read_segmentation_dir,
    read_image_dir,
    seg_to_coords,
    image_to_coords,
    quantify_coords,
    save_analysis,
    interpolate_volume,
    save_volumes,
)


class AnalysisWorker(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, arguments: Dict[str, Any]):
        super().__init__()
        self.arguments = arguments
        self.cancelled = False

    def run(self):
        sys.stdout = TextRedirector()
        sys.stderr = sys.stdout
        sys.stdout.text_written.connect(self.log_signal.emit)
        try:
            print("Starting analysis... This may take a moment")

            if self.cancelled:
                print("Analysis cancelled")
                return

            # Load atlas
            if self.arguments.get("use_custom_atlas", False):
                print(
                    f"Using custom atlas: {self.arguments.get('custom_atlas_name', 'Custom')}"
                )
                atlas = load_custom_atlas(
                    self.arguments["atlas_path"],
                    self.arguments.get("hemi_path"),
                    self.arguments["label_path"],
                )
            else:
                print(f"Using BrainGlobe atlas: {self.arguments['atlas_name']}")
                atlas = resolve_atlas(BrainGlobeAtlas(self.arguments["atlas_name"]))

            if self.cancelled:
                print("Analysis cancelled")
                return

            # Load registration
            alignment_json = self.arguments["registration_json"]
            apply_damage_mask = self.arguments["apply_damage_mask"]
            registration = read_alignment(
                alignment_json, apply_damage=apply_damage_mask
            )

            # Extract coordinates
            seg_dir = self.arguments.get("segmentation_dir")
            img_dir = self.arguments.get("image_dir")
            seg_format = self.arguments.get("segmentation_format", "binary")

            if img_dir and not seg_dir:
                result = image_to_coords(
                    img_dir,
                    registration,
                    atlas,
                )
            else:
                result = seg_to_coords(
                    seg_dir,
                    registration,
                    atlas,
                    pixel_id=self.arguments["object_colour"],
                    segmentation_format=seg_format,
                )

            if self.cancelled:
                print("Analysis cancelled")
                return

            label_df = quantify_coords(result, atlas.labels)

            if self.cancelled:
                print("Analysis cancelled")
                return

            volumes = {}
            if self.arguments.get("interpolate_volume"):
                value_mode = self.arguments.get("value_mode", "pixel_count")
                print(f"Creating interpolated volume (mode: {value_mode})...")
                if seg_dir:
                    vol_series = read_segmentation_dir(
                        seg_dir,
                        pixel_id=self.arguments["object_colour"],
                        segmentation_format=seg_format,
                    )
                else:
                    vol_series = read_image_dir(img_dir)
                volumes = interpolate_volume(
                    image_series=vol_series,
                    registration=registration,
                    atlas=atlas,
                    value_mode=value_mode,
                    segmentation_mode=bool(seg_dir),
                )

            if self.cancelled:
                print("Analysis cancelled")
                return

            output_dir = self.arguments["output_dir"]
            save_analysis(
                output_dir,
                result,
                atlas.labels,
                label_df=label_df,
            )
            if volumes:
                save_volumes(
                    output_folder=output_dir,
                    volumes=volumes,
                    atlas=atlas,
                )

            print(f"Analysis complete. Results saved to {output_dir}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def cancel(self):
        self.cancelled = True
        print("Cancellation requested, please wait...")


class AtlasInstallWorker(QThread):
    """Worker thread for installing BrainGlobe atlases."""

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # Success flag and message

    def __init__(self, atlas_name: str):
        super().__init__()
        self.atlas_name = atlas_name
        self.cancelled = False

    def run(self):
        sys.stdout = TextRedirector()
        sys.stderr = sys.stdout
        sys.stdout.text_written.connect(self.progress_signal.emit)

        try:
            self.progress_signal.emit(f"Starting installation of {self.atlas_name}...")
            BrainGlobeAtlas(self.atlas_name)

            if self.cancelled:
                self.finished_signal.emit(
                    False, f"Installation of {self.atlas_name} was cancelled."
                )
            else:
                self.finished_signal.emit(
                    True,
                    f"BrainGlobe atlas '{self.atlas_name}' installed successfully.",
                )

        except Exception as e:
            error_msg = f"Error installing BrainGlobe atlas: {str(e)}"
            self.progress_signal.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def cancel(self):
        self.cancelled = True

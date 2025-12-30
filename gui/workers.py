import io
import sys
from typing import Any, Dict

import brainglobe_atlasapi
from PyQt6.QtCore import QObject, QThread, pyqtSignal

import PyNutil


class TextRedirector(QObject):
    text_written = pyqtSignal(str)

    def write(self, text: str):
        if text.strip():
            self.text_written.emit(text)

    def flush(self):
        pass


class AnalysisWorker(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, arguments: Dict[str, Any]):
        super().__init__()
        self.arguments = arguments
        self.cancelled = False

    def run(self):
        buffer = io.StringIO()
        sys.stdout = TextRedirector()
        sys.stdout.text_written.connect(self.log_signal.emit)
        try:
            print("Starting analysis... This may take a moment")

            if self.cancelled:
                print("Analysis cancelled")
                return

            pnt_args = {
                "segmentation_folder": self.arguments["segmentation_dir"],
                "alignment_json": self.arguments["registration_json"],
                "colour": self.arguments["object_colour"],
                "custom_region_path": self.arguments.get("custom_region_path"),
            }

            if self.arguments.get("use_custom_atlas", False):
                pnt_args["atlas_path"] = self.arguments["atlas_path"]
                pnt_args["label_path"] = self.arguments["label_path"]
                print(
                    f"Using custom atlas: {self.arguments.get('custom_atlas_name', 'Custom')}"
                )
            else:
                pnt_args["atlas_name"] = self.arguments["atlas_name"]
                print(f"Using BrainGlobe atlas: {self.arguments['atlas_name']}")

            pnt = PyNutil.PyNutil(**pnt_args)

            if self.cancelled:
                print("Analysis cancelled")
                return

            pnt.get_coordinates(
                object_cutoff=0, apply_damage_mask=self.arguments["apply_damage_mask"]
            )

            if self.cancelled:
                print("Analysis cancelled")
                return

            pnt.quantify_coordinates()

            if self.cancelled:
                print("Analysis cancelled")
                return

            pnt.save_analysis(self.arguments["output_dir"])
        except Exception as e:
            print(f"Error: {e}")
        finally:
            sys.stdout = sys.__stdout__

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
        sys.stdout.text_written.connect(self.progress_signal.emit)

        try:
            self.progress_signal.emit(f"Starting installation of {self.atlas_name}...")
            brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas(self.atlas_name)

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

    def cancel(self):
        self.cancelled = True

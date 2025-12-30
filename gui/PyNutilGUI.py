# Import restructuring
import sys
import json
import os
from typing import Dict, List, Optional, Union, Any

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QFileDialog,
    QColorDialog,
    QComboBox,
    QMenuBar,
    QTextBrowser,
    QPlainTextEdit,
    QMessageBox,
    QDialog,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QGridLayout,
    QGroupBox,
    QProgressDialog,
    QCheckBox,
)
from PyQt6.QtGui import QAction, QColor, QIcon, QDesktopServices
from PyQt6.QtCore import (
    QMetaObject,
    Qt,
    Q_ARG,
    QUrl,
    pyqtSlot,
)

import brainglobe_atlasapi
import brainglobe_atlasapi.utils as bg_utils
import PyNutil

from workers import AnalysisWorker, AtlasInstallWorker
from validation import validate_analysis_inputs

# Import UI component utilities
from ui_components import (
    create_labeled_combo_with_button,
    create_horizontal_combo_with_button,
    get_path_display_name,
    populate_dropdown,
    create_atlas_installation_dialog,
    create_run_buttons_layout,
    select_path,
    create_path_selection_section,
)

#  Patch retrieve_over_http to update progress via GUI's update_progress slot
original_retrieve = bg_utils.retrieve_over_http


def patched_retrieve_over_http(url, output_file_path, fn_update=None):
    # Try to extract atlas name from URL or path
    atlas_name = "Atlas"
    # URLs usually have this format: http://example.com/path/atlas_name/file.nii.gz
    # So we can try to extract the atlas name from the URL parts
    url_parts = url.split("/")
    if len(url_parts) >= 2:
        potential_atlas_name = url_parts[-2]
        if potential_atlas_name and not potential_atlas_name.startswith("http"):
            atlas_name = potential_atlas_name

    # Signal the start of atlas download with the atlas name
    if hasattr(PyNutilGUI, "instance") and PyNutilGUI.instance:
        QMetaObject.invokeMethod(
            PyNutilGUI.instance,
            "atlas_download_started",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, atlas_name),
        )

    def patched_fn_update(completed, total):
        if total > 0:
            percent = completed / total
            total_mb = total / (1000 * 1000)  # Convert bytes to MB
            current_mb = completed / (1000 * 1000)
            percent_int = int(round(percent * 100))  # Convert to integer percentage
        else:
            percent = 0
            total_mb = 0
            current_mb = 0
            percent_int = 0

        # Create progress text - no need to check for dialog visibility
        # since we'll always use the dialog
        progress_text = f"Downloading atlas ({current_mb:.2f} / {total_mb:.2f} MB)..."

        if hasattr(PyNutilGUI, "instance") and PyNutilGUI.instance:
            # Always send the signal to update the UI with the dialog
            QMetaObject.invokeMethod(
                PyNutilGUI.instance,
                "update_download_progress",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, progress_text),
                Q_ARG(int, percent_int),
            )

            # Check if download should be cancelled
            if (
                hasattr(PyNutilGUI.instance, "download_cancelled")
                and PyNutilGUI.instance.download_cancelled
            ):
                raise Exception("Download cancelled by user")

        if fn_update:
            fn_update(completed, total)

    try:
        result = original_retrieve(url, output_file_path, fn_update=patched_fn_update)
        # Signal the end of the download if successful
        if hasattr(PyNutilGUI, "instance") and PyNutilGUI.instance:
            QMetaObject.invokeMethod(
                PyNutilGUI.instance,
                "atlas_download_finished",
                Qt.ConnectionType.QueuedConnection,
            )
        return result
    except Exception as e:
        # Signal the end of download even if there was an exception
        if hasattr(PyNutilGUI, "instance") and PyNutilGUI.instance:
            QMetaObject.invokeMethod(
                PyNutilGUI.instance,
                "atlas_download_finished",
                Qt.ConnectionType.QueuedConnection,
            )
        # Re-raise the exception
        raise e


bg_utils.retrieve_over_http = patched_retrieve_over_http


class PyNutilGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        PyNutilGUI.instance = self  # store the instance for progress updates
        self.setWindowTitle("PyNutil")

        # Add flags to track download state
        self.download_cancelled = False  # Track if user requested cancellation
        self.is_downloading_atlas = False  # Track if atlas download is in progress
        self.current_atlas_name = (
            None  # Track the name of the atlas currently being downloaded
        )

        # Set the application icon
        icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "Logo_PyNutil.ico"
        )
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Warning: Icon file not found at {icon_path}")

        self.arguments = {
            "reference_atlas": None,
            "registration_json": None,
            "object_colour": None,
            "segmentation_dir": None,
            "output_dir": None,
            "label_path": None,  # Added for custom atlases
            "atlas_path": None,  # Added for custom atlases
            "custom_region_path": None,  # Added for custom region file (optional)
            "apply_damage_mask": False,  # <-- ADDED default field
        }
        self.recent_files_path = os.path.join(
            os.path.expanduser("~"), ".pynutil_recent_files.json"
        )
        self.recent_files = self.load_recent_files()
        if "custom_region" not in self.recent_files:
            self.recent_files["custom_region"] = []
        self.initUI()

    def load_recent_files(self):
        if not os.path.exists(self.recent_files_path):
            data = {
                "registration_json": [""],
                "segmentation_dir": [""],
                "output_dir": [""],
                "object_colour": [""],
                "custom_atlases": [],
                "custom_region": [""],
            }
            with open(self.recent_files_path, "w") as file:
                json.dump(data, file)
            return data
        with open(self.recent_files_path, "r") as file:
            data = json.load(file)
            for key in [
                "registration_json",
                "segmentation_dir",
                "output_dir",
                "object_colour",
                "custom_region",
            ]:
                if not isinstance(data.get(key, []), list):
                    data[key] = []
                # Ensure there is one default empty item if the list is empty
                if len(data.get(key, [])) == 0:
                    data[key] = [""]
            if "custom_atlases" not in data:
                data["custom_atlases"] = []
            return data

    def save_recent_files(self):
        with open(self.recent_files_path, "w") as file:
            json.dump(self.recent_files, file)

    def initUI(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create left panel
        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setLayout(left_layout)

        # Create menu bar
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("File")
        help_menu = menubar.addMenu("Help")

        load_settings_action = QAction("Load Settings", self)
        load_settings_action.triggered.connect(self.load_settings_from_file)
        file_menu.addAction(load_settings_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        about_action = QAction("About PyNutil", self)
        about_action.triggered.connect(self.about_pynutil)
        help_menu.addAction(about_action)

        self.setMenuBar(menubar)

        # Atlas selection - use a more compact approach
        atlas_layout = QVBoxLayout()
        atlas_layout.addWidget(QLabel("Select reference atlas:"))

        # Create horizontal layout for atlas dropdown and install button
        atlas_h_layout = QHBoxLayout()
        self.atlas_combo = QComboBox()
        self.atlas_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        self.populate_atlas_dropdown()
        self.atlas_combo.setCurrentIndex(-1)

        self.install_atlas_button = QPushButton("+")
        self.install_atlas_button.setToolTip("Install or Add Atlas")
        self.install_atlas_button.setMaximumWidth(30)
        self.install_atlas_button.clicked.connect(self.show_install_atlas_dialog)

        atlas_h_layout.addWidget(self.atlas_combo, 1)
        atlas_h_layout.addWidget(self.install_atlas_button, 0)

        # Add the horizontal layout to the vertical layout
        atlas_layout.addLayout(atlas_h_layout)

        # Add to the main left layout with minimal margin
        left_layout.addLayout(atlas_layout)
        # left_layout.setSpacing(10)  # Control spacing between sections

        # Registration JSON selection using unified path selection function
        registration_layout, self.registration_json_dropdown, _ = (
            create_path_selection_section(
                parent=self,
                label_text="Select registration JSON:",
                path_type="file",
                title="Open Registration JSON",
                key="registration_json",
                recents=self.recent_files["registration_json"],
                callback=self.set_registration_json,
                argument_dict=self.arguments,
            )
        )
        left_layout.addLayout(registration_layout)

        # Segmentation directory selection using unified path selection function
        segmentation_layout, self.segmentation_dir_dropdown, _ = (
            create_path_selection_section(
                parent=self,
                label_text="Select segmentation folder:",
                path_type="directory",
                title="Select Segmentation Directory",
                key="segmentation_dir",
                recents=self.recent_files["segmentation_dir"],
                callback=self.set_segmentation_dir,
                argument_dict=self.arguments,
            )
        )
        left_layout.addLayout(segmentation_layout)

        # Object color selection
        color_layout, self.colour_dropdown, self.colour_button = (
            create_labeled_combo_with_button(
                "Select object colour:",
                button_text="Colour",
                button_callback=self.choose_colour,
            )
        )
        populate_dropdown(
            self.colour_dropdown, self.recent_files.get("object_colour", [])
        )
        self.colour_dropdown.currentIndexChanged.connect(self.set_colour)
        left_layout.addLayout(color_layout)

        # Custom region file selection (optional)
        custom_region_layout, self.custom_region_dropdown, _ = (
            create_path_selection_section(
                parent=self,
                label_text="Select custom region file (optional):",
                path_type="file",
                title="Select Custom Region File",
                key="custom_region",
                recents=self.recent_files.get("custom_region", []),
                callback=self.set_custom_region_file,
                argument_dict=self.arguments,
            )
        )
        left_layout.addLayout(custom_region_layout)

        # Output directory selection using unified path selection function
        output_layout, self.output_dir_dropdown, _ = create_path_selection_section(
            parent=self,
            label_text="Select output directory:",
            path_type="directory",
            title="Select Output Directory",
            key="output_dir",
            recents=self.recent_files["output_dir"],
            callback=self.set_output_dir,
            argument_dict=self.arguments,
        )
        left_layout.addLayout(output_layout)

        damage_markers_layout = QHBoxLayout()
        damage_markers_label = QLabel("Include Damage Quantification:")
        self.include_damage_markers_checkbox = QCheckBox()
        self.include_damage_markers_checkbox.setChecked(False)
        self.include_damage_markers_checkbox.stateChanged.connect(
            self.update_damage_markers_flag
        )

        damage_markers_layout.addWidget(damage_markers_label)
        damage_markers_layout.addWidget(self.include_damage_markers_checkbox)
        left_layout.addLayout(damage_markers_layout)

        # Run and cancel buttons
        left_layout.addWidget(QLabel("Start analysis:"))
        run_buttons_layout, self.run_button, self.cancel_button = (
            create_run_buttons_layout()
        )
        self.run_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        left_layout.addLayout(run_buttons_layout)

        # Create right panel with output browser
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        output_label = QLabel("Output:")
        right_layout.addWidget(output_label)

        # Output text browser
        self.output_box = QTextBrowser()
        self.output_box.setOpenExternalLinks(True)
        self.output_box.setMinimumWidth(600)
        self.output_box.setMinimumHeight(400)
        right_layout.addWidget(self.output_box)

        # Add panels to main layout
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.setMinimumSize(1000, 600)

        # Initialize log storage variables
        self.log_collection = ""
        self.current_progress = ""

    def donothing(self):
        pass

    def about_pynutil(self):
        help_text = """PyNutil is a Python library for brain-wide quantification and spatial analysis of features in serial section images from mouse and rat brain. It aims to replicate the Quantifier feature of the Nutil software (RRID: SCR_017183). It builds on registration to a standardised reference atlas with the QuickNII (RRID:SCR_016854) and VisuAlign software (RRID:SCR_017978) and feature extraction by segmentation with an image analysis software such as ilastik (RRID:SCR_015246).

For more information about the QUINT workflow: <a href="https://quint-workflow.readthedocs.io/en/latest/">https://quint-workflow.readthedocs.io/en/latest/</a>"""

        self.output_box.clear()
        self.output_box.setHtml(help_text.replace("\n", "<br>"))

    def set_registration_json(self, index):
        if index >= 0:
            # Retrieve the full path from userData
            value = (
                self.registration_json_dropdown.itemData(index)
                or self.registration_json_dropdown.currentText()
            )
            self.arguments["registration_json"] = value

    def set_segmentation_dir(self, index):
        if index >= 0:
            value = (
                self.segmentation_dir_dropdown.itemData(index)
                or self.segmentation_dir_dropdown.currentText()
            )
            self.arguments["segmentation_dir"] = value

    def set_output_dir(self, index):
        if index >= 0:
            value = (
                self.output_dir_dropdown.itemData(index)
                or self.output_dir_dropdown.currentText()
            )
            self.arguments["output_dir"] = value

    def set_custom_region_file(self, index):
        if index >= 0:
            value = (
                self.custom_region_dropdown.itemData(index)
                or self.custom_region_dropdown.currentText()
            )
            self.arguments["custom_region_path"] = value

    def set_colour(self, index):
        if index >= 0:
            value = (
                self.colour_dropdown.itemData(index)
                or self.colour_dropdown.currentText()
            )
            if value:
                try:
                    rgb_str = value.strip("[]")
                    rgb_values = [int(x.strip()) for x in rgb_str.split(",")]
                    self.arguments["object_colour"] = rgb_values
                except Exception:
                    self.arguments["object_colour"] = [0, 0, 0]

    def update_recent(self, key, value):
        if not value.strip():
            return  # ignore empty values
        recents = self.recent_files.get(key, [])
        # Remove any default empty entries
        recents = [entry for entry in recents if entry.strip()]
        if value in recents:
            recents.remove(value)
        recents.insert(0, value)
        self.recent_files[key] = recents[:5]
        self.save_recent_files()

    # We still need the color chooser since it's different from file/directory selection
    def choose_colour(self):
        value = QColorDialog.getColor()
        if value.isValid():
            rgb_list = [value.red(), value.green(), value.blue()]
            self.arguments["object_colour"] = rgb_list
            rgb_str = f"[{value.red()}, {value.green()}, {value.blue()}]"
            self.update_recent("object_colour", rgb_str)
            populate_dropdown(
                self.colour_dropdown, self.recent_files.get("object_colour", [])
            )
            # For color items, store the RGB string in the user data too
            self.colour_dropdown.setItemData(1, rgb_str)
            self.colour_dropdown.setCurrentIndex(1)

    def start_analysis(self):
        # Clear both the output box and the log collection variables
        self.output_box.clear()
        self.log_collection = ""
        self.current_progress = ""

        # Get atlas information - check if it's a custom atlas (stored as userData)
        atlas_name = self.atlas_combo.currentText()
        custom_atlas_data = self.atlas_combo.currentData()

        validation = validate_analysis_inputs(
            atlas_text=atlas_name,
            arguments=self.arguments,
            custom_atlas_data=custom_atlas_data,
        )
        if not validation.ok:
            self.output_box.setHtml(validation.to_html())
            return

        # Prepare arguments for the worker
        if custom_atlas_data:
            # It's a custom atlas, set up the custom atlas paths
            self.arguments["atlas_path"] = custom_atlas_data["atlas_path"]
            self.arguments["label_path"] = custom_atlas_data["label_path"]
            self.arguments["use_custom_atlas"] = True
            self.arguments["custom_atlas_name"] = custom_atlas_data["name"]
            # We still need to pass atlas_name as None for backward compatibility
            self.arguments["atlas_name"] = None
        else:
            # It's a standard atlas, clear any custom paths
            self.arguments["atlas_path"] = None
            self.arguments["label_path"] = None
            self.arguments["use_custom_atlas"] = False
            self.arguments["atlas_name"] = atlas_name

        # Include custom region file (optional)
        # If no custom region file is selected, it remains None.
        # Prepare worker parameters
        pnt_args = {
            "segmentation_folder": self.arguments["segmentation_dir"],
            "alignment_json": self.arguments["registration_json"],
            "colour": self.arguments["object_colour"],
            "custom_region_path": self.arguments.get("custom_region_path"),
        }
        # If all validations pass, start the worker
        self.worker = AnalysisWorker(self.arguments)
        # Disable the run button until analysis finishes
        self.run_button.setEnabled(False)
        # Enable the cancel button
        self.cancel_button.setEnabled(True)
        # Re-enable the run button when the worker finishes
        self.worker.finished.connect(self.analysis_finished)
        self.worker.log_signal.connect(self.append_text_to_output)
        self.worker.start()

    def analysis_finished(self):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def cancel_analysis(self):
        """Handle cancellation of analysis or download"""
        # Cancel analysis if a worker thread is running
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.cancel()
            # UI buttons will be updated via the finished signal

        # Use the generalized cancel_download method for download cancellations
        if self.is_downloading_atlas:
            self.cancel_download()

    def remove_lines_from_log(self, num_lines=1, pattern=None):
        """
        Remove lines from the log collection.

        Args:
            num_lines: Number of lines to remove from the end of the log
            pattern: If provided, remove lines containing this pattern
        """
        if not self.log_collection:
            return

        lines = self.log_collection.rstrip("<br>").split("<br>")

        if pattern:
            # Remove lines containing the pattern (searching from the end)
            new_lines = []
            for line in lines:
                if pattern not in line:
                    new_lines.append(line)
            lines = new_lines
        elif num_lines > 0:
            # Remove the specified number of lines from the end
            lines = lines[:-num_lines] if num_lines < len(lines) else []

        self.log_collection = "<br>".join(lines) + ("<br>" if lines else "")
        # Update display
        self.output_box.setHtml(self.log_collection + self.current_progress)

    def append_text_to_output(self, text):
        # Use the new utility function instead of inline line removal
        if "Downloading Atlas File" in text:
            self.remove_lines_from_log(pattern="Downloading Atlas File")
        elif "Atlas download cancelled" in text:
            self.remove_lines_from_log(num_lines=3)

        # Append the new text
        self.log_collection += text.replace("\n", "<br>") + "<br>"
        self.output_box.setHtml(self.log_collection + self.current_progress)
        sb = self.output_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def load_settings_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as file:
                settings = json.load(file)

            if "segmentation_folder" in settings and settings["segmentation_folder"]:
                self.arguments["segmentation_dir"] = settings["segmentation_folder"]
                self.update_recent("segmentation_dir", settings["segmentation_folder"])
                populate_dropdown(
                    self.segmentation_dir_dropdown,
                    self.recent_files["segmentation_dir"],
                )
                self.segmentation_dir_dropdown.setCurrentIndex(0)

            if "alignment_json" in settings and settings["alignment_json"]:
                self.arguments["registration_json"] = settings["alignment_json"]
                self.update_recent("registration_json", settings["alignment_json"])
                populate_dropdown(
                    self.registration_json_dropdown,
                    self.recent_files["registration_json"],
                )
                # Set index 0 to correctly select the loaded alignment JSON.
                self.registration_json_dropdown.setCurrentIndex(0)

            if "colour" in settings and settings["colour"]:
                rgb_list = settings["colour"]
                self.arguments["object_colour"] = rgb_list
                rgb_str = f"[{rgb_list[0]}, {rgb_list[1]}, {rgb_list[2]}]"
                self.update_recent("object_colour", rgb_str)
                populate_dropdown(
                    self.colour_dropdown, self.recent_files.get("object_colour", [])
                )
                self.colour_dropdown.setCurrentIndex(1)

            if "custom_region" in settings and settings["custom_region"]:
                self.arguments["custom_region_path"] = settings["custom_region"]
                self.update_recent("custom_region", settings["custom_region"])
                populate_dropdown(
                    self.custom_region_dropdown,
                    self.recent_files.get("custom_region", []),
                )
                self.custom_region_dropdown.setCurrentIndex(1)

            if "atlas_name" in settings and settings["atlas_name"]:
                atlas_name = settings["atlas_name"]
                index = self.atlas_combo.findText(atlas_name)
                if index >= 0:
                    self.atlas_combo.setCurrentIndex(index)

            if "include_damage_markers" in settings:
                dmg_markers = bool(settings["include_damage_markers"])
                self.arguments["apply_damage_mask"] = dmg_markers
                self.include_damage_markers_checkbox.setChecked(dmg_markers)

            self.output_box.setHtml(f"Settings loaded from: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load settings: {str(e)}")
            self.output_box.setHtml(f"Error loading settings: {str(e)}")

    @pyqtSlot(str)
    def update_progress(self, text: str):
        self.current_progress = text
        self.output_box.setHtml(self.log_collection + self.current_progress)

    @pyqtSlot(str, int)
    def update_download_progress(self, text: str, percent: int):
        """Update the progress dialog during atlas download"""
        # Always ensure progress dialog exists and is visible
        if not (
            hasattr(self, "progress_dialog")
            and self.progress_dialog
            and self.progress_dialog.isVisible()
        ):
            # If the dialog isn't visible, create it or show it
            self.ensure_progress_dialog_visible()

        # Now update the progress dialog
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(text)

    def ensure_progress_dialog_visible(self):
        """Make sure the progress dialog is created and visible"""
        if not hasattr(self, "progress_dialog") or not self.progress_dialog:
            # Create a new progress dialog
            self.progress_dialog = QProgressDialog(
                "Downloading atlas...", "Cancel", 0, 100, self
            )
            self.progress_dialog.setWindowTitle("Atlas Download")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setMinimumWidth(400)
            self.progress_dialog.setValue(0)

        # Always set up a new cancel button to ensure proper connection
        # First, remove any existing cancel button by setting it to None
        self.progress_dialog.setCancelButton(None)

        # Create a new button and connect it
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_download)
        self.progress_dialog.setCancelButton(cancel_button)

        self.progress_dialog.closeEvent = self.handle_progress_dialog_close

        if not self.progress_dialog.isVisible():
            self.progress_dialog.show()

    def handle_progress_dialog_close(self, event):
        if self.is_downloading_atlas:
            self.cancel_download()
            event.accept()

    @pyqtSlot(str)
    def atlas_download_started(self, atlas_name: str):
        """Called when atlas download begins"""
        self.is_downloading_atlas = True
        self.download_cancelled = False
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        # Use atlas_name from the worker if available, otherwise use the provided one
        display_name = self.current_atlas_name or atlas_name

        # Always ensure the progress dialog is visible
        self.ensure_progress_dialog_visible()
        self.progress_dialog.setWindowTitle(f"Downloading Atlas: {display_name}")

    @pyqtSlot()
    def atlas_download_finished(self):
        """Called when atlas download completes or is cancelled"""
        self.is_downloading_atlas = False

        # Close any progress dialog that might still be visible
        if (
            hasattr(self, "progress_dialog")
            and self.progress_dialog
            and self.progress_dialog.isVisible()
        ):
            self.progress_dialog.close()

        # Only re-enable run button if there's no analysis running
        if not hasattr(self, "worker") or not self.worker.isRunning():
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

        # Update the user about what happened
        if self.download_cancelled:
            self.append_text_to_output("Atlas download cancelled.")
        else:
            # Use the utility function to remove download status lines
            self.remove_lines_from_log(num_lines=2)
            self.append_text_to_output("Atlas download finished.")

    def populate_atlas_dropdown(self):
        self.atlas_combo.clear()
        added_empty = False
        # Add installed BrainGlobe atlases
        for atlas in brainglobe_atlasapi.list_atlases.get_atlases_lastversions():
            if atlas == "":
                if not added_empty:
                    self.atlas_combo.addItem(atlas)
                    added_empty = True
            else:
                self.atlas_combo.addItem(atlas)
        # Add custom atlases from recent files
        for atlas in self.recent_files.get("custom_atlases", []):
            name = atlas.get("name", "")
            if name == "":
                pass
            else:
                self.atlas_combo.addItem(name, userData=atlas)

    def show_install_atlas_dialog(self):
        """Show a dialog to install a new atlas."""
        (
            dialog,
            brain_globe_radio,
            custom_radio,
            brain_globe_group,
            custom_group,
            self.brain_globe_combo,
            install_brain_globe_button,
            self.custom_atlas_name_edit,
            self.custom_atlas_path_edit,
            self.custom_label_path_edit,
            browse_atlas_button,
            browse_label_button,
            add_custom_button,
        ) = create_atlas_installation_dialog(self)

        # Populate brainglobe atlas combo
        available_atlases = (
            brainglobe_atlasapi.list_atlases.get_all_atlases_lastversions()
        )
        self.brain_globe_combo.addItems(available_atlases)

        # Connect buttons to functions
        install_brain_globe_button.clicked.connect(self.install_brain_globe_atlas)
        browse_atlas_button.clicked.connect(self.browse_custom_atlas_path)
        browse_label_button.clicked.connect(self.browse_custom_label_path)
        add_custom_button.clicked.connect(self.add_custom_atlas)

        dialog.exec()

    def install_brain_globe_atlas(self):
        """Install the selected BrainGlobe atlas with progress monitoring"""
        atlas_name = self.brain_globe_combo.currentText()
        if not atlas_name:
            QMessageBox.warning(
                self, "Warning", "Please select a BrainGlobe atlas to install."
            )
            return

        # Store the current atlas name so it can be used in download dialogs
        self.current_atlas_name = atlas_name

        # Create progress dialog with range 0-100 for percentage
        self.progress_dialog = QProgressDialog(
            f"Installing atlas: {atlas_name}...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle(f"Installing Atlas: {atlas_name}")
        self.progress_dialog.setModal(True)
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setValue(0)  # Start at 0%
        self.progress_dialog.setCancelButton(None)  # Hide the default cancel button

        # Add custom cancel button with generalized cancel method
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(
            self.cancel_download
        )  # Use generalized cancel method
        self.progress_dialog.setCancelButton(cancel_button)

        # Create and start worker thread
        self.install_worker = AtlasInstallWorker(atlas_name)
        self.install_worker.progress_signal.connect(self.update_atlas_install_progress)
        self.install_worker.finished_signal.connect(self.atlas_installation_complete)

        self.progress_dialog.show()
        self.install_worker.start()

    def update_atlas_install_progress(self, message):
        """Update the progress dialog with installation status"""
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setLabelText(message)
            # Keep dialog responsive
            QApplication.processEvents()
            # Don't output to main log when dialog is visible
        else:
            # Only output to main log when dialog is not visible (for silent updates)
            self.append_text_to_output(message)

    def atlas_installation_complete(self, success, message):
        """Handle atlas installation completion"""
        # Close progress dialog
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.close()

        # Handle result
        if success:
            self.append_text_to_output(message)
            self.populate_atlas_dropdown()
        else:
            QMessageBox.critical(self, "Error", message)
            self.append_text_to_output(message)

    def cancel_download(self):
        """General method to cancel any ongoing download"""
        self.download_cancelled = True
        self.append_text_to_output("Cancelling download. Please wait...")

        # Close the progress dialog immediately
        if (
            hasattr(self, "progress_dialog")
            and self.progress_dialog
            and self.progress_dialog.isVisible()
        ):
            self.progress_dialog.close()

        # Also cancel specific workers if available
        if hasattr(self, "install_worker") and self.install_worker.isRunning():
            self.install_worker.cancel()

    def browse_custom_atlas_path(self):
        """Browse for custom atlas path."""
        path = select_path(
            parent=self, path_type="file", title="Select Custom Atlas File"
        )
        if path:
            self.custom_atlas_path_edit.setText(path)

    def browse_custom_label_path(self):
        """Browse for custom label path."""
        path = select_path(
            parent=self, path_type="file", title="Select Custom Label File"
        )
        if path:
            self.custom_label_path_edit.setText(path)

    def add_custom_atlas(self):
        """Add a custom atlas to the recent files and update the dropdown."""
        atlas_name = self.custom_atlas_name_edit.text().strip()
        atlas_path = self.custom_atlas_path_edit.text().strip()
        label_path = self.custom_label_path_edit.text().strip()

        if not atlas_name or not atlas_path or not label_path:
            QMessageBox.warning(
                self, "Warning", "Please provide all fields for the custom atlas."
            )
            return

        # Create custom atlas dictionary
        custom_atlas = {
            "name": atlas_name,
            "atlas_path": atlas_path,
            "label_path": label_path,
        }

        # Check if atlas with same name already exists
        existing_atlases = self.recent_files.get("custom_atlases", [])
        for i, atlas in enumerate(existing_atlases):
            if atlas.get("name") == atlas_name:
                # Replace existing entry
                existing_atlases[i] = custom_atlas
                self.save_recent_files()
                self.populate_atlas_dropdown()
                self.append_text_to_output(
                    f"Custom atlas '{atlas_name}' updated successfully."
                )
                return

        # Add new custom atlas if not updating an existing one
        if "custom_atlases" not in self.recent_files:
            self.recent_files["custom_atlases"] = []
        self.recent_files["custom_atlases"].append(custom_atlas)
        self.save_recent_files()
        self.populate_atlas_dropdown()
        self.append_text_to_output(f"Custom atlas '{atlas_name}' added successfully.")

    def update_damage_markers_flag(self, state):
        self.arguments["apply_damage_mask"] = bool(state)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PyNutilGUI()
    gui.show()
    sys.exit(app.exec())

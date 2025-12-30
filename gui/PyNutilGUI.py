# Import restructuring
import sys
import json
import os

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QColorDialog,
    QMenuBar,
    QTextBrowser,
    QMessageBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QSizePolicy,
)
from PyQt6.QtGui import QAction,  QIcon


import brainglobe_atlasapi

from workers import AnalysisWorker, AtlasInstallWorker
from validation import validate_analysis_inputs

# Import UI component utilities
from ui_components import (
    create_labeled_combo_with_button,
    create_horizontal_combo_with_button,
    populate_dropdown,
    CappedPopupComboBox,
    AtlasInstallationDialog,
    create_run_buttons_layout,
    select_path,
    create_path_selection_section,
)

from settings_manager import SettingsManager
from log_manager import LogManager


class PyNutilGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyNutil")

        # Initialize managers
        settings_path = os.path.join(os.path.expanduser("~"), ".pynutil_recent_files.json")
        self.settings_manager = SettingsManager(settings_path)
        self.recent_files = self.settings_manager.settings

        # Add flags to track download state
        self.download_cancelled = False
        self.is_downloading_atlas = False
        self.current_atlas_name = None

        # Set the application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logo_PyNutil.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.arguments = {
            "reference_atlas": None,
            "registration_json": None,
            "object_colour": "auto",
            "segmentation_dir": None,
            "image_dir": None,
            "output_dir": None,
            "label_path": None,
            "atlas_path": None,
            "custom_region_path": None,
            "apply_damage_mask": False,
            "cellpose": False,
            "interpolate_volume": False,
            "value_mode": "pixel_count",
        }
        self.initUI()
        self.log_manager = LogManager(self.output_box)

    def update_recent(self, key, value):
        self.settings_manager.update_recent(key, value)
        self.recent_files = self.settings_manager.settings

    def initUI(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create left panel
        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        # Prefer flexible sizing over large fixed minimums to avoid overlapping on small windows
        left_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

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

        # Atlas selection
        atlas_layout = QVBoxLayout()
        atlas_layout.addWidget(QLabel("Select reference atlas:"))
        atlas_h_layout, self.atlas_combo, self.install_atlas_button = (
            create_horizontal_combo_with_button(parent=self)
        )
        self.populate_atlas_dropdown()
        self.atlas_combo.setCurrentIndex(-1)
        self.install_atlas_button.clicked.connect(self.show_install_atlas_dialog)
        atlas_layout.addLayout(atlas_h_layout)
        left_layout.addLayout(atlas_layout)

        # Registration JSON selection
        registration_layout, self.registration_json_dropdown, _ = (
            create_path_selection_section(
                parent=self,
                label_text="Select registration JSON:",
                key="registration_json",
                recents=self.recent_files.get("registration_json", []),
                callback=self.set_registration_json,
            )
        )
        left_layout.addLayout(registration_layout)

        # Input mode selector (mutually exclusive) - either segmentation or image
        mode_group = QGroupBox("Input type")
        mode_layout = QHBoxLayout()
        self.seg_radio = QRadioButton("Segmentation folder")
        self.img_radio = QRadioButton("Image folder (intensity)")
        self.seg_radio.setChecked(True)
        mode_layout.addWidget(self.seg_radio)
        mode_layout.addWidget(self.img_radio)
        mode_group.setLayout(mode_layout)
        self.seg_radio.toggled.connect(self.set_input_mode_segmentation)
        self.img_radio.toggled.connect(self.set_input_mode_image)
        left_layout.addWidget(mode_group)

        # Segmentation directory selection (visible when segmentation mode selected)
        segmentation_layout, self.segmentation_dir_dropdown, self.segmentation_dir_button = (
            create_path_selection_section(
                parent=self,
                label_text="Select segmentation folder:",
                path_type="directory",
                key="segmentation_dir",
                recents=self.recent_files.get("segmentation_dir", []),
                callback=self.set_segmentation_dir,
            )
        )
        self.segmentation_widget = QWidget()
        self.segmentation_widget.setLayout(segmentation_layout)
        left_layout.addWidget(self.segmentation_widget)

        # Image directory selection (visible when image mode selected)
        image_layout, self.image_dir_dropdown, self.image_dir_button = (
            create_path_selection_section(
                parent=self,
                label_text="Select image folder (for intensity):",
                path_type="directory",
                key="image_dir",
                recents=self.recent_files.get("image_dir", []),
                callback=self.set_image_dir,
            )
        )
        self.image_widget = QWidget()
        self.image_widget.setLayout(image_layout)
        left_layout.addWidget(self.image_widget)

        # By default show segmentation widget and hide image widget
        self.segmentation_widget.setVisible(True)
        self.image_widget.setVisible(False)

        # Cellpose checkbox
        cellpose_layout = QHBoxLayout()
        cellpose_label = QLabel("Cellpose segmentation:")
        self.cellpose_checkbox = QCheckBox()
        self.cellpose_checkbox.setChecked(False)
        self.cellpose_checkbox.stateChanged.connect(self.update_cellpose_flag)
        cellpose_layout.addWidget(cellpose_label)
        cellpose_layout.addWidget(self.cellpose_checkbox)
        left_layout.addLayout(cellpose_layout)

        # Object color selection
        color_layout, self.colour_dropdown, self.colour_button = (
            create_labeled_combo_with_button(
                "Select object colour:",
                button_text="Colour",
                button_callback=self.choose_colour,
                parent=self,
            )
        )
        # Allow manual entry (e.g. "auto", "0", "1", or "[r,g,b]")
        self.colour_dropdown.setEditable(True)
        populate_dropdown(
            self.colour_dropdown, self.recent_files.get("object_colour", [])
        )
        # Add an explicit "auto" option at the top (populate clears, so insert after)
        self.colour_dropdown.insertItem(0, "auto", userData="auto")
        # Default to auto
        self.colour_dropdown.setCurrentIndex(0)
        self.colour_dropdown.currentIndexChanged.connect(self.set_colour)
        left_layout.addLayout(color_layout)

        # Custom region file selection
        custom_region_layout, self.custom_region_dropdown, _ = (
            create_path_selection_section(
                parent=self,
                label_text="Select custom region file (optional):",
                key="custom_region",
                recents=self.recent_files.get("custom_region", []),
                callback=self.set_custom_region_file,
            )
        )
        left_layout.addLayout(custom_region_layout)

        # Output directory selection
        output_layout, self.output_dir_dropdown, _ = create_path_selection_section(
            parent=self,
            label_text="Select output directory:",
            path_type="directory",
            key="output_dir",
            recents=self.recent_files.get("output_dir", []),
            callback=self.set_output_dir,
        )
        left_layout.addLayout(output_layout)

        # Interpolated volume options
        interpolate_layout = QVBoxLayout()
        interpolate_check_layout = QHBoxLayout()
        interpolate_label = QLabel("Create interpolated volume:")
        self.interpolate_checkbox = QCheckBox()
        self.interpolate_checkbox.setChecked(False)
        self.interpolate_checkbox.stateChanged.connect(self.update_interpolate_flag)
        interpolate_check_layout.addWidget(interpolate_label)
        interpolate_check_layout.addWidget(self.interpolate_checkbox)
        interpolate_layout.addLayout(interpolate_check_layout)

        value_mode_layout = QHBoxLayout()
        value_mode_label = QLabel("Value mode:")
        self.value_mode_combo = CappedPopupComboBox(self)
        self.value_mode_combo.addItems(["pixel_count", "mean", "object_count"])
        self.value_mode_combo.setEnabled(False)
        self.value_mode_combo.currentTextChanged.connect(self.set_value_mode)
        value_mode_layout.addWidget(value_mode_label)
        value_mode_layout.addWidget(self.value_mode_combo)
        interpolate_layout.addLayout(value_mode_layout)
        left_layout.addLayout(interpolate_layout)

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
        # Allow the output box to shrink reasonably on small windows
        self.output_box.setMinimumWidth(300)
        self.output_box.setMinimumHeight(300)
        self.output_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.output_box)

        # Add panels to main layout
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Reduce strict minimum size to allow resizing on Windows without overlap
        self.setMinimumSize(800, 500)

        # Initialize log storage variables
        self.log_collection = ""
        self.current_progress = ""

    def about_pynutil(self):
        help_text = """PyNutil is a Python library for brain-wide quantification and spatial analysis of features in serial section images from mouse and rat brain. It aims to replicate the Quantifier feature of the Nutil software (RRID: SCR_017183). It builds on registration to a standardised reference atlas with the QuickNII (RRID:SCR_016854) and VisuAlign software (RRID:SCR_017978) and feature extraction by segmentation with an image analysis software such as ilastik (RRID:SCR_015246).

For more information about the QUINT workflow: <a href="https://quint-workflow.readthedocs.io/en/latest/">https://quint-workflow.readthedocs.io/en/latest/</a>"""

        self.log_manager.clear()
        self.log_manager.append(help_text)

    def set_registration_json(self, index):
        if index >= 0:
            self.arguments["registration_json"] = self.registration_json_dropdown.itemData(index) or self.registration_json_dropdown.currentText()

    def set_segmentation_dir(self, index):
        if index >= 0:
            self.arguments["segmentation_dir"] = self.segmentation_dir_dropdown.itemData(index) or self.segmentation_dir_dropdown.currentText()

    def set_image_dir(self, index):
        if index >= 0:
            self.arguments["image_dir"] = self.image_dir_dropdown.itemData(index) or self.image_dir_dropdown.currentText()

    def update_cellpose_flag(self, state):
        is_cellpose = bool(state)
        self.arguments["cellpose"] = is_cellpose
        # Disable colour controls when cellpose is selected
        try:
            self.colour_dropdown.setEnabled(not is_cellpose)
        except Exception:
            pass
        try:
            self.colour_button.setEnabled(not is_cellpose)
        except Exception:
            pass
        if is_cellpose:
            # Clear any previously selected colour to avoid conflict
            self.arguments["object_colour"] = None
            try:
                # reset dropdown to empty selection
                self.colour_dropdown.setCurrentIndex(0)
            except Exception:
                pass

    def update_interpolate_flag(self, state):
        is_checked = bool(state)
        self.arguments["interpolate_volume"] = is_checked
        self.value_mode_combo.setEnabled(is_checked)

    def set_value_mode(self, text):
        self.arguments["value_mode"] = text

    def set_input_mode_segmentation(self, checked: bool):
        # When segmentation mode is selected, show segmentation widget and hide image widget
        if checked:
            try:
                self.segmentation_widget.setVisible(True)
            except Exception:
                pass
            try:
                self.image_widget.setVisible(False)
            except Exception:
                pass
            # clear any image_dir argument
            self.arguments["image_dir"] = None

    def set_input_mode_image(self, checked: bool):
        # When image mode is selected, show image widget and hide segmentation widget
        if checked:
            try:
                self.image_widget.setVisible(True)
            except Exception:
                pass
            try:
                self.segmentation_widget.setVisible(False)
            except Exception:
                pass
            # clear any segmentation_dir argument
            self.arguments["segmentation_dir"] = None

    def set_output_dir(self, index):
        if index >= 0:
            self.arguments["output_dir"] = self.output_dir_dropdown.itemData(index) or self.output_dir_dropdown.currentText()

    def set_custom_region_file(self, index):
        if index >= 0:
            self.arguments["custom_region_path"] = self.custom_region_dropdown.itemData(index) or self.custom_region_dropdown.currentText()

    def set_colour(self, index):
        if index >= 0:
            raw = self.colour_dropdown.itemData(index)
            if raw is None:
                raw = self.colour_dropdown.currentText()
            value = raw
            if value is None:
                return
            # Accept explicit 'auto'
            if isinstance(value, str) and value.strip().lower() == "auto":
                self.arguments["object_colour"] = "auto"
                return
            # If it's a stored string representation like '[r, g, b]'
            if isinstance(value, str):
                s = value.strip()
                # single integer like '0' or '1'
                if s.isdigit():
                    try:
                        self.arguments["object_colour"] = [int(s)]
                        return
                    except Exception:
                        pass
                # comma separated numbers (either 'r,g,b' or 'r, g, b')
                if ("," in s) or (s.startswith("[") and s.endswith("]")):
                    try:
                        nums = [int(x.strip()) for x in s.strip("[]").split(",") if x.strip()]
                        self.arguments["object_colour"] = nums
                        return
                    except Exception:
                        pass
            # If it's already a list/iterable
            try:
                # convert to list of ints
                lst = list(value)
                lst = [int(x) for x in lst]
                self.arguments["object_colour"] = lst
                return
            except Exception:
                # fallback
                self.arguments["object_colour"] = "auto"

    def choose_colour(self):
        color = QColorDialog.getColor()
        if color.isValid():
            rgb_list = [color.red(), color.green(), color.blue()]
            self.arguments["object_colour"] = rgb_list
            rgb_str = str(rgb_list)
            self.update_recent("object_colour", rgb_str)
            populate_dropdown(self.colour_dropdown, self.recent_files.get("object_colour", []))
            # re-insert auto at position 0 after populate
            try:
                self.colour_dropdown.insertItem(0, "auto", userData="auto")
            except Exception:
                pass
            # set to the newly added colour (search for its text)
            idx = self.colour_dropdown.findText(rgb_str)
            if idx >= 0:
                self.colour_dropdown.setCurrentIndex(idx)

    def start_analysis(self):
        # Clear logs
        self.log_manager.clear()

        # Get atlas information
        atlas_name = self.atlas_combo.currentText()
        custom_atlas_data = self.atlas_combo.currentData()

        validation = validate_analysis_inputs(
            atlas_text=atlas_name,
            arguments=self.arguments,
            custom_atlas_data=custom_atlas_data,
        )
        if not validation.ok:
            self.log_manager.append(validation.to_html())
            return

        # Prepare arguments for the worker
        if custom_atlas_data:
            self.arguments.update({
                "atlas_path": custom_atlas_data["atlas_path"],
                "label_path": custom_atlas_data["label_path"],
                "use_custom_atlas": True,
                "custom_atlas_name": custom_atlas_data["name"],
                "atlas_name": None
            })
        else:
            self.arguments.update({
                "atlas_path": None,
                "label_path": None,
                "use_custom_atlas": False,
                "atlas_name": atlas_name
            })

        # Update recent files
        if self.arguments["registration_json"]:
            self.update_recent("registration_json", self.arguments["registration_json"])
        if self.arguments["segmentation_dir"]:
            self.update_recent("segmentation_dir", self.arguments["segmentation_dir"])
        if self.arguments["image_dir"]:
            self.update_recent("image_dir", self.arguments["image_dir"])
        if self.arguments["output_dir"]:
            self.update_recent("output_dir", self.arguments["output_dir"])
        if self.arguments["custom_region_path"]:
            self.update_recent("custom_region", self.arguments["custom_region_path"])

        self.worker = AnalysisWorker(self.arguments)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
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

    def append_text_to_output(self, text):
        self.log_manager.append(text)

    def load_settings_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as file:
                settings = json.load(file)

            if settings.get("segmentation_folder"):
                self.arguments["segmentation_dir"] = settings["segmentation_folder"]
                self.update_recent("segmentation_dir", settings["segmentation_folder"])
                populate_dropdown(self.segmentation_dir_dropdown, self.recent_files["segmentation_dir"])
                self.segmentation_dir_dropdown.setCurrentIndex(0)

            if settings.get("image_folder"):
                self.arguments["image_dir"] = settings["image_folder"]
                self.update_recent("image_dir", settings["image_folder"])
                populate_dropdown(self.image_dir_dropdown, self.recent_files["image_dir"])
                self.image_dir_dropdown.setCurrentIndex(0)

            # Set input mode radio and visibility based on loaded settings
            seg_set = bool(settings.get("segmentation_folder"))
            img_set = bool(settings.get("image_folder"))
            if seg_set and not img_set:
                self.seg_radio.setChecked(True)
                try:
                    self.segmentation_widget.setVisible(True)
                    self.image_widget.setVisible(False)
                except Exception:
                    pass
            elif img_set and not seg_set:
                self.img_radio.setChecked(True)
                try:
                    self.image_widget.setVisible(True)
                    self.segmentation_widget.setVisible(False)
                except Exception:
                    pass
            elif seg_set and img_set:
                # Both set in file: prefer segmentation and warn
                self.seg_radio.setChecked(True)
                try:
                    self.segmentation_widget.setVisible(True)
                    self.image_widget.setVisible(False)
                except Exception:
                    pass
                self.log_manager.append("Warning: both segmentation and image folders present in settings; using segmentation folder.")

            if settings.get("alignment_json"):
                self.arguments["registration_json"] = settings["alignment_json"]
                self.update_recent("registration_json", settings["alignment_json"])
                populate_dropdown(self.registration_json_dropdown, self.recent_files["registration_json"])
                self.registration_json_dropdown.setCurrentIndex(0)

            if settings.get("colour") is not None:
                self.arguments["object_colour"] = settings["colour"]
                rgb_str = str(settings["colour"])
                self.update_recent("object_colour", rgb_str)
                populate_dropdown(self.colour_dropdown, self.recent_files.get("object_colour", []))
                try:
                    # ensure auto option present
                    self.colour_dropdown.insertItem(0, "auto", userData="auto")
                except Exception:
                    pass
                # Try to select the loaded colour, otherwise leave as auto
                idx = self.colour_dropdown.findText(rgb_str)
                if idx >= 0:
                    self.colour_dropdown.setCurrentIndex(idx)
                else:
                    self.colour_dropdown.setCurrentIndex(0)

            if settings.get("custom_region"):
                self.arguments["custom_region_path"] = settings["custom_region"]
                self.update_recent("custom_region", settings["custom_region"])
                populate_dropdown(self.custom_region_dropdown, self.recent_files.get("custom_region", []))
                self.custom_region_dropdown.setCurrentIndex(1)

            if settings.get("atlas_name"):
                index = self.atlas_combo.findText(settings["atlas_name"])
                if index >= 0:
                    self.atlas_combo.setCurrentIndex(index)

            if "include_damage_markers" in settings:
                dmg_markers = bool(settings["include_damage_markers"])
                self.arguments["apply_damage_mask"] = dmg_markers
                self.include_damage_markers_checkbox.setChecked(dmg_markers)

            if "cellpose" in settings:
                self.cellpose_checkbox.setChecked(bool(settings["cellpose"]))

            if "interpolate_volume" in settings:
                self.interpolate_checkbox.setChecked(bool(settings["interpolate_volume"]))

            if settings.get("value_mode"):
                index = self.value_mode_combo.findText(settings["value_mode"])
                if index >= 0:
                    self.value_mode_combo.setCurrentIndex(index)

            # If both cellpose and colour were present in settings, prefer cellpose and clear colour
            if settings.get("cellpose") and settings.get("colour"):
                self.arguments["object_colour"] = None
                try:
                    self.colour_dropdown.setCurrentIndex(0)
                    self.colour_dropdown.setEnabled(False)
                    self.colour_button.setEnabled(False)
                except Exception:
                    pass
                self.log_manager.append("Warning: settings specified both 'cellpose' and 'colour'; using cellpose and clearing colour.")

            self.log_manager.append(f"Settings loaded from: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load settings: {str(e)}")
            self.log_manager.append(f"Error loading settings: {str(e)}")

    def populate_atlas_dropdown(self):
        self.atlas_combo.clear()
        self.atlas_combo.addItem("")
        # Add BrainGlobe atlases
        for atlas in brainglobe_atlasapi.list_atlases.get_atlases_lastversions():
            if atlas:
                self.atlas_combo.addItem(atlas)
        # Add custom atlases
        for atlas in self.recent_files.get("custom_atlases", []):
            if atlas.get("name"):
                self.atlas_combo.addItem(atlas["name"], userData=atlas)

    def show_install_atlas_dialog(self):
        """Show a dialog to install a new atlas."""
        dialog = AtlasInstallationDialog(self)

        # Populate brainglobe atlas combo
        available_atlases = brainglobe_atlasapi.list_atlases.get_all_atlases_lastversions()
        dialog.brain_globe_combo.addItems(available_atlases)

        # Connect buttons
        dialog.install_button.clicked.connect(lambda: self.install_brain_globe_atlas(dialog.brain_globe_combo.currentText()))
        dialog.browse_path_btn.clicked.connect(lambda: self.browse_custom_path(dialog.path_edit))
        dialog.browse_label_btn.clicked.connect(lambda: self.browse_custom_path(dialog.label_edit))
        dialog.add_custom_btn.clicked.connect(lambda: self.add_custom_atlas(dialog))

        dialog.exec()

    def browse_custom_path(self, line_edit):
        path = select_path(parent=self, path_type="file")
        if path:
            line_edit.setText(path)

    def install_brain_globe_atlas(self, atlas_name):
        """Install the selected BrainGlobe atlas"""
        if not atlas_name:
            QMessageBox.warning(self, "Warning", "Please select a BrainGlobe atlas to install.")
            return

        self.current_atlas_name = atlas_name
        self.install_worker = AtlasInstallWorker(atlas_name)
        self.install_worker.progress_signal.connect(self.append_text_to_output)
        self.install_worker.finished_signal.connect(self.atlas_installation_complete)
        self.install_worker.start()

    def atlas_installation_complete(self, success, message):
        """Handle atlas installation completion"""
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
        if hasattr(self, "install_worker") and self.install_worker.isRunning():
            self.install_worker.cancel()

    def add_custom_atlas(self, dialog):
        """Add a custom atlas to the recent files."""
        name = dialog.name_edit.text().strip()
        path = dialog.path_edit.text().strip()
        label = dialog.label_edit.text().strip()

        if not all([name, path, label]):
            QMessageBox.warning(self, "Warning", "Please provide all fields.")
            return

        custom_atlas = {"name": name, "atlas_path": path, "label_path": label}

        # Update custom atlases in settings
        atlases = self.recent_files.get("custom_atlases", [])
        # Remove if exists
        atlases = [a for a in atlases if a.get("name") != name]
        atlases.append(custom_atlas)
        self.recent_files["custom_atlases"] = atlases
        self.settings_manager.save_settings()

        self.populate_atlas_dropdown()
        self.append_text_to_output(f"Custom atlas '{name}' added successfully.")
        dialog.accept()

    def update_damage_markers_flag(self, state):
        self.arguments["apply_damage_mask"] = bool(state)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PyNutilGUI()
    gui.show()
    sys.exit(app.exec())

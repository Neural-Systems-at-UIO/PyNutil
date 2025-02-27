import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QFileDialog, QColorDialog, QComboBox, QMenuBar, QTextBrowser, QPlainTextEdit, QMessageBox, QTextBrowser
)
from PyQt6.QtGui import QAction, QColor, QIcon
from PyQt6.QtCore import QMetaObject, Qt, Q_ARG, QThread, pyqtSignal, QObject, QUrl
# Add this import
from PyQt6.QtGui import QDesktopServices

import brainglobe_atlasapi
import PyNutil
import threading
import io
import contextlib
import json
import os

class TextRedirector(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # Only emit non-empty texts
            self.text_written.emit(text)

    def flush(self):
        pass

class AnalysisWorker(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, arguments, atlas_name):
        super().__init__()
        self.arguments = arguments
        self.atlas_name = atlas_name

    def run(self):
        buffer = io.StringIO()
        sys.stdout = TextRedirector()
        sys.stdout.text_written.connect(self.log_signal.emit)
        try:
            pnt = PyNutil.PyNutil(
                segmentation_folder=self.arguments["segmentation_dir"],
                alignment_json=self.arguments["registration_json"],
                colour=self.arguments["object_colour"],
                atlas_name=self.atlas_name,
            )

            pnt.get_coordinates(object_cutoff=0)
            pnt.quantify_coordinates()
            pnt.save_analysis(self.arguments["output_dir"])
        except Exception as e:
            print(f"Error: {e}")
        finally:
            sys.stdout = sys.__stdout__

class PyNutilGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyNutil")

        # Set the application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logo_PyNutil.ico")
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
        }
        self.recent_files_path = os.path.join(os.path.expanduser("~"), ".pynutil_recent_files.json")
        self.recent_files = self.load_recent_files()
        self.initUI()

    def load_recent_files(self):
        if (os.path.exists(self.recent_files_path)):
            with open(self.recent_files_path, "r") as file:
                data = json.load(file)
                for key in ["registration_json", "segmentation_dir", "output_dir"]:
                    if not isinstance(data.get(key, []), list):
                        data[key] = [data.get(key)] if data.get(key) else []
                if "object_colour" not in data:
                    data["object_colour"] = []
                return data
        return {
            "registration_json": [],
            "segmentation_dir": [],
            "output_dir": [],
            "object_colour": []
        }

    def save_recent_files(self):
        with open(self.recent_files_path, "w") as file:
            json.dump(self.recent_files, file)

    def get_path_display_name(self, path):
        """Extract the filename or last directory from a path for display."""
        if not path:
            return ""
        if os.path.isfile(path) or path.endswith((".json", ".txt")):
            return os.path.basename(path)
        else:
            # For directories, show the last directory name
            path = path.rstrip(os.path.sep)  # Remove trailing slashes
            return os.path.basename(path)

    def populate_dropdown(self, dropdown, recents):
        dropdown.clear()
        dropdown.addItem("")

        # Store the full paths as user data but show shortened displays
        for item in recents:
            display_text = self.get_path_display_name(item)
            dropdown.addItem(display_text)
            # Store the full path as user data in the item
            dropdown.setItemData(dropdown.count() - 1, item)

        dropdown.setEditable(False)
        dropdown.setCurrentIndex(-1)

    def initUI(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setLayout(left_layout)

        menubar = QMenuBar(self)
        file_menu = menubar.addMenu('File')
        help_menu = menubar.addMenu('Help')
        load_settings_action = QAction('Load Settings', self)
        load_settings_action.triggered.connect(self.load_settings_from_file)
        file_menu.addAction(load_settings_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        about_action = QAction('About PyNutil', self)
        about_action.triggered.connect(self.about_pynutil)
        help_menu.addAction(about_action)

        self.setMenuBar(menubar)

        left_layout.addWidget(QLabel("Select reference atlas:"))
        self.atlas_combo = QComboBox()
        self.atlas_combo.setStyleSheet("QComboBox { combobox-popup: 0; }");
        atlas = brainglobe_atlasapi.list_atlases.get_all_atlases_lastversions()
        self.atlas_combo.addItems(atlas)
        self.atlas_combo.setCurrentIndex(-1)
        left_layout.addWidget(self.atlas_combo)

        left_layout.addWidget(QLabel("Select registration JSON:"))
        self.registration_json_button = QPushButton("Browse...")
        self.registration_json_button.clicked.connect(self.open_registration_json)
        self.registration_json_dropdown = QComboBox()
        self.registration_json_dropdown.setStyleSheet("QComboBox { combobox-popup: 0; }");

        self.populate_dropdown(self.registration_json_dropdown, self.recent_files["registration_json"])
        self.registration_json_dropdown.currentIndexChanged.connect(self.set_registration_json)
        left_layout.addWidget(self.registration_json_button)
        left_layout.addWidget(self.registration_json_dropdown)

        left_layout.addWidget(QLabel("Select segmentation folder:"))
        self.segmentation_dir_button = QPushButton("Browse...")
        self.segmentation_dir_button.clicked.connect(self.open_segmentation_dir)
        self.segmentation_dir_dropdown = QComboBox()
        self.segmentation_dir_dropdown.setStyleSheet("QComboBox { combobox-popup: 0; }");

        self.populate_dropdown(self.segmentation_dir_dropdown, self.recent_files["segmentation_dir"])
        self.segmentation_dir_dropdown.currentIndexChanged.connect(self.set_segmentation_dir)
        left_layout.addWidget(self.segmentation_dir_button)
        left_layout.addWidget(self.segmentation_dir_dropdown)

        left_layout.addWidget(QLabel("Select object colour:"))
        self.colour_button = QPushButton("Colour")
        self.colour_button.clicked.connect(self.choose_colour)
        left_layout.addWidget(self.colour_button)
        self.colour_dropdown = QComboBox()
        self.colour_dropdown.setStyleSheet("QComboBox { combobox-popup: 0; }");
        self.populate_dropdown(self.colour_dropdown, self.recent_files.get("object_colour", []))
        self.colour_dropdown.currentIndexChanged.connect(self.set_colour)
        left_layout.addWidget(self.colour_dropdown)

        left_layout.addWidget(QLabel("Select output directory:"))
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        self.output_dir_dropdown = QComboBox()
        self.colour_dropdown.setStyleSheet("QComboBox { combobox-popup: 0; }");
        self.populate_dropdown(self.output_dir_dropdown, self.recent_files["output_dir"])
        self.output_dir_dropdown.currentIndexChanged.connect(self.set_output_dir)
        left_layout.addWidget(self.output_dir_button)
        left_layout.addWidget(self.output_dir_dropdown)

        left_layout.addWidget(QLabel("Start analysis:"))
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.start_analysis)
        left_layout.addWidget(self.run_button)

        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        output_label = QLabel("Output:")
        right_layout.addWidget(output_label)

        # Replace QTextEdit with QTextBrowser
        self.output_box = QTextBrowser()
        self.output_box.setOpenExternalLinks(True)  # This works with QTextBrowser
        self.output_box.setMinimumWidth(600)
        self.output_box.setMinimumHeight(400)
        right_layout.addWidget(self.output_box)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.setMinimumSize(1000, 600)

    def donothing(self):
        pass

    def about_pynutil(self):
        help_text = """PyNutil is a Python library for brain-wide quantification and spatial analysis of features in serial section images from mouse and rat brain. It aims to replicate the Quantifier feature of the Nutil software (RRID: SCR_017183). It builds on registration to a standardised reference atlas with the QuickNII (RRID:SCR_016854) and VisuAlign software (RRID:SCR_017978) and feature extraction by segmentation with an image analysis software such as ilastik (RRID:SCR_015246).

For more information about the QUINT workflow: <a href="https://quint-workflow.readthedocs.io/en/latest/">https://quint-workflow.readthedocs.io/en/latest/</a>"""

        self.output_box.clear()
        self.output_box.setHtml(help_text.replace("\n", "<br>"))
        # These settings are no longer needed as QTextBrowser handles them automatically
        # self.output_box.document().setDefaultStyleSheet("a { color: blue; text-decoration: underline; }")
        # self.output_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)

    def set_registration_json(self, index):
        if index >= 0:
            # Get the full path from the user data, not the display text
            value = self.registration_json_dropdown.itemData(index)
            self.arguments["registration_json"] = value

    def set_segmentation_dir(self, index):
        if index >= 0:
            value = self.segmentation_dir_dropdown.itemData(index)
            self.arguments["segmentation_dir"] = value

    def set_output_dir(self, index):
        if index >= 0:
            value = self.output_dir_dropdown.itemData(index)
            self.arguments["output_dir"] = value

    def set_colour(self, index):
        if index >= 0:
            value = self.colour_dropdown.itemData(index) or self.colour_dropdown.itemText(index)
            if value:
                try:
                    rgb_str = value.strip("[]")
                    rgb_values = [int(x.strip()) for x in rgb_str.split(",")]
                    self.arguments["object_colour"] = rgb_values
                except:
                    self.arguments["object_colour"] = [0, 0, 0]

    def update_recent(self, key, value):
        recents = self.recent_files.get(key, [])
        if value in recents:
            recents.remove(value)
        recents.insert(0, value)
        self.recent_files[key] = recents[:5]
        self.save_recent_files()

    def open_registration_json(self):
        value, _ = QFileDialog.getOpenFileName(self, "Open Registration JSON")
        if value:
            self.arguments["registration_json"] = value
            self.update_recent("registration_json", value)
            self.populate_dropdown(self.registration_json_dropdown, self.recent_files["registration_json"])
            self.registration_json_dropdown.setCurrentIndex(1)

    def choose_colour(self):
        value = QColorDialog.getColor()
        if value.isValid():
            rgb_list = [value.red(), value.green(), value.blue()]
            self.arguments["object_colour"] = rgb_list
            rgb_str = f"[{value.red()}, {value.green()}, {value.blue()}]"
            self.update_recent("object_colour", rgb_str)
            self.populate_dropdown(self.colour_dropdown, self.recent_files.get("object_colour", []))
            # For color items, store the RGB string in the user data too
            self.colour_dropdown.setItemData(1, rgb_str)
            self.colour_dropdown.setCurrentIndex(1)

    def open_segmentation_dir(self):
        value = QFileDialog.getExistingDirectory(self, "Select Segmentation Directory")
        if value:
            self.arguments["segmentation_dir"] = value
            self.update_recent("segmentation_dir", value)
            self.populate_dropdown(self.segmentation_dir_dropdown, self.recent_files["segmentation_dir"])
            self.segmentation_dir_dropdown.setCurrentIndex(1)

    def select_output_dir(self):
        value = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if value:
            self.arguments["output_dir"] = value
            self.update_recent("output_dir", value)
            self.populate_dropdown(self.output_dir_dropdown, self.recent_files["output_dir"])
            self.output_dir_dropdown.setCurrentIndex(1)

    def start_analysis(self):
        self.output_box.clear()

        # Validate all required settings are provided
        missing_settings = []

        if not self.atlas_combo.currentText():
            missing_settings.append("Reference Atlas")

        if not self.arguments.get("registration_json"):
            missing_settings.append("Registration JSON")

        if not self.arguments.get("segmentation_dir"):
            missing_settings.append("Segmentation Folder")

        if not self.arguments.get("object_colour"):
            missing_settings.append("Object Color")

        if not self.arguments.get("output_dir"):
            missing_settings.append("Output Directory")

        if missing_settings:
            error_message = "Error: The following required settings are missing:<br>"
            for setting in missing_settings:
                error_message += f"- {setting}<br>"
            error_message += "<br>Please provide all required settings before running the analysis."
            self.output_box.setHtml(error_message)
            return

        # If all validations pass, start the worker
        self.worker = AnalysisWorker(self.arguments, self.atlas_combo.currentText())
        # We need to modify how messages are appended since we're now using HTML
        self.worker.log_signal.connect(self.append_text_to_output)
        self.worker.start()

    def append_text_to_output(self, text):
        """Append text to the output box as HTML."""
        # QTextBrowser appends work a little differently
        self.output_box.append(text.replace("\n", "<br>"))
        # Ensure we scroll to the bottom after appending
        sb = self.output_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def load_settings_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as file:
                settings = json.load(file)

            if "segmentation_folder" in settings and settings["segmentation_folder"]:
                self.arguments["segmentation_dir"] = settings["segmentation_folder"]
                self.update_recent("segmentation_dir", settings["segmentation_folder"])
                self.populate_dropdown(self.segmentation_dir_dropdown, self.recent_files["segmentation_dir"])
                self.segmentation_dir_dropdown.setCurrentIndex(1)

            if "alignment_json" in settings and settings["alignment_json"]:
                self.arguments["registration_json"] = settings["alignment_json"]
                self.update_recent("registration_json", settings["alignment_json"])
                self.populate_dropdown(self.registration_json_dropdown, self.recent_files["registration_json"])
                self.registration_json_dropdown.setCurrentIndex(1)

            if "colour" in settings and settings["colour"]:
                rgb_list = settings["colour"]
                self.arguments["object_colour"] = rgb_list
                rgb_str = f"[{rgb_list[0]}, {rgb_list[1]}, {rgb_list[2]}]"
                self.update_recent("object_colour", rgb_str)
                self.populate_dropdown(self.colour_dropdown, self.recent_files.get("object_colour", []))
                self.colour_dropdown.setCurrentIndex(1)

            if "atlas_name" in settings and settings["atlas_name"]:
                atlas_name = settings["atlas_name"]
                index = self.atlas_combo.findText(atlas_name)
                if index >= 0:
                    self.atlas_combo.setCurrentIndex(index)

            self.output_box.setHtml(f"Settings loaded from: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load settings: {str(e)}")
            self.output_box.setHtml(f"Error loading settings: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PyNutilGUI()
    gui.show()
    sys.exit(app.exec())

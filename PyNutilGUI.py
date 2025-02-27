import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QFileDialog, QColorDialog, QComboBox, QMenuBar, QTextEdit, QPlainTextEdit
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QMetaObject, Qt, Q_ARG, QThread
import brainglobe_atlasapi
import PyNutil
import threading
import io
import contextlib
import json
import os

class PyNutilGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyNutil")
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
        if os.path.exists(self.recent_files_path):
            with open(self.recent_files_path, "r") as file:
                data = json.load(file)
                # Ensure each recent is a list
                for key in ["registration_json", "segmentation_dir", "output_dir"]:
                    if not isinstance(data.get(key, []), list):
                        data[key] = [data.get(key)] if data.get(key) else []
                return data
        return {
            "registration_json": [],
            "segmentation_dir": [],
            "output_dir": []
        }

    def save_recent_files(self):
        with open(self.recent_files_path, "w") as file:
            json.dump(self.recent_files, file)

    def populate_dropdown(self, dropdown, recents):
        dropdown.clear()
        # Add empty item first so nothing is selected by default
        dropdown.addItem("")
        for item in recents:
            dropdown.addItem(item)
        # Set to non-editable (default behavior)
        dropdown.setEditable(False)
        # Make sure no item is selected by default
        dropdown.setCurrentIndex(-1)

    def initUI(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Menu bar
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu('File')
        help_menu = menubar.addMenu('Help')

        new_action = QAction('New', self)
        new_action.triggered.connect(self.donothing)
        file_menu.addAction(new_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        about_action = QAction('About PyNutil', self)
        about_action.triggered.connect(self.about_pynutil)
        help_menu.addAction(about_action)

        self.setMenuBar(menubar)

        # Select reference atlas
        left_layout.addWidget(QLabel("Select reference atlas:"))
        self.atlas_combo = QComboBox()
        atlas = brainglobe_atlasapi.list_atlases.get_all_atlases_lastversions()
        self.atlas_combo.addItems(atlas)
        # Set no selection by default
        self.atlas_combo.setCurrentIndex(-1)
        left_layout.addWidget(self.atlas_combo)

        # Select registration JSON
        left_layout.addWidget(QLabel("Select registration JSON:"))
        self.registration_json_button = QPushButton("Browse...")
        self.registration_json_button.clicked.connect(self.open_registration_json)
        self.registration_json_dropdown = QComboBox()
        self.populate_dropdown(self.registration_json_dropdown, self.recent_files["registration_json"])
        self.registration_json_dropdown.currentIndexChanged.connect(self.set_registration_json)
        left_layout.addWidget(self.registration_json_button)
        left_layout.addWidget(self.registration_json_dropdown)

        # Select segmentation folder
        left_layout.addWidget(QLabel("Select segmentation folder:"))
        self.segmentation_dir_button = QPushButton("Browse...")
        self.segmentation_dir_button.clicked.connect(self.open_segmentation_dir)
        self.segmentation_dir_dropdown = QComboBox()
        self.populate_dropdown(self.segmentation_dir_dropdown, self.recent_files["segmentation_dir"])
        self.segmentation_dir_dropdown.currentIndexChanged.connect(self.set_segmentation_dir)
        left_layout.addWidget(self.segmentation_dir_button)
        left_layout.addWidget(self.segmentation_dir_dropdown)

        # Select object colour
        left_layout.addWidget(QLabel("Select object colour:"))
        self.colour_button = QPushButton("Colour")
        self.colour_button.clicked.connect(self.choose_colour)
        left_layout.addWidget(self.colour_button)
        self.colour_dropdown = QComboBox()
        self.populate_dropdown(self.colour_dropdown, self.recent_files.get("object_colour", []))
        self.colour_dropdown.currentIndexChanged.connect(self.set_colour)
        left_layout.addWidget(self.colour_dropdown)

        # Select output directory
        left_layout.addWidget(QLabel("Select output directory:"))
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        self.output_dir_dropdown = QComboBox()
        self.populate_dropdown(self.output_dir_dropdown, self.recent_files["output_dir"])
        self.output_dir_dropdown.currentIndexChanged.connect(self.set_output_dir)
        left_layout.addWidget(self.output_dir_button)
        left_layout.addWidget(self.output_dir_dropdown)

        # Start analysis
        left_layout.addWidget(QLabel("Start analysis:"))
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.start_analysis)
        left_layout.addWidget(self.run_button)

        # Output box
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Output:"))
        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        right_layout.addWidget(self.output_box)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def donothing(self):
        pass

    def about_pynutil(self):
        about_msg = QLabel("PyNutil is an application for brain-wide mapping using a reference brain atlas")
        about_msg.show()

    def set_registration_json(self, index):
        if index >= 0:
            value = self.registration_json_dropdown.itemText(index)
            self.arguments["registration_json"] = value

    def set_segmentation_dir(self, index):
        if index >= 0:
            value = self.segmentation_dir_dropdown.itemText(index)
            self.arguments["segmentation_dir"] = value

    def set_output_dir(self, index):
        if index >= 0:
            value = self.output_dir_dropdown.itemText(index)
            self.arguments["output_dir"] = value

    def set_colour(self, index):
        if index >= 0:
            value = self.colour_dropdown.itemText(index)
            self.arguments["object_colour"] = value

    def update_recent(self, key, value):
        # Insert new entry at the beginning, remove duplicates
        recents = self.recent_files.get(key, [])
        if value in recents:
            recents.remove(value)
        recents.insert(0, value)
        # Optional: Limit length
        self.recent_files[key] = recents[:5]
        self.save_recent_files()

    def open_registration_json(self):
        value, _ = QFileDialog.getOpenFileName(self, "Open Registration JSON")
        if value:
            self.arguments["registration_json"] = value
            self.update_recent("registration_json", value)
            self.populate_dropdown(self.registration_json_dropdown, self.recent_files["registration_json"])
            # Set the dropdown to select the new item (index 1, after the empty item)
            self.registration_json_dropdown.setCurrentIndex(1)

    def choose_colour(self):
        value = QColorDialog.getColor()
        if value.isValid():
            self.arguments["object_colour"] = value.name()
            self.update_recent("object_colour", value.name())
            self.populate_dropdown(self.colour_dropdown, self.recent_files.get("object_colour", []))
            # Set the dropdown to select the new item
            self.colour_dropdown.setCurrentIndex(1)

    def open_segmentation_dir(self):
        value = QFileDialog.getExistingDirectory(self, "Select Segmentation Directory")
        if value:
            self.arguments["segmentation_dir"] = value
            self.update_recent("segmentation_dir", value)
            self.populate_dropdown(self.segmentation_dir_dropdown, self.recent_files["segmentation_dir"])
            # Set the dropdown to select the new item
            self.segmentation_dir_dropdown.setCurrentIndex(1)

    def select_output_dir(self):
        value = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if value:
            self.arguments["output_dir"] = value
            self.update_recent("output_dir", value)
            self.populate_dropdown(self.output_dir_dropdown, self.recent_files["output_dir"])
            # Set the dropdown to select the new item
            self.output_dir_dropdown.setCurrentIndex(1)

    def start_analysis(self):
        self.output_box.clear()
        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def run_analysis(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                pnt = PyNutil.PyNutil(
                    segmentation_folder=self.arguments["segmentation_dir"],
                    alignment_json=self.arguments["registration_json"],
                    colour=self.arguments["object_colour"],
                    atlas_name=self.atlas_combo.currentText(),
                )

                pnt.get_coordinates(object_cutoff=0)
                pnt.quantify_coordinates()
                pnt.save_analysis(self.arguments["output_dir"])
            except Exception as e:
                print(f"Error: {e}")
        output = buffer.getvalue()
        QMetaObject.invokeMethod(self.output_box, "setPlainText", Qt.ConnectionType.QueuedConnection, Q_ARG(str, output))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PyNutilGUI()
    gui.show()
    sys.exit(app.exec())

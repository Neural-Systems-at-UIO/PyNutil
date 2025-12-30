"""
UI Component utilities for PyNutil GUI
"""

from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QDialog,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QLineEdit,
    QGridLayout,
)
import os


class CappedPopupComboBox(QComboBox):
    """A QComboBox with a limit on visible items."""
    def __init__(self, parent=None, max_visible_items: int = 20):
        super().__init__(parent)
        self.setMaxVisibleItems(max_visible_items)


def create_labeled_combo_with_button(
    label_text: str,
    button_text: str = "Browse...",
    button_callback=None,
    combo_callback=None,
    parent=None,
) -> tuple:
    """Create a standard vertical layout with label, button, and combo box."""
    layout = QVBoxLayout()
    layout.addWidget(QLabel(label_text))

    button = QPushButton(button_text, parent)
    if button_callback:
        button.clicked.connect(button_callback)
    layout.addWidget(button)

    combo_box = CappedPopupComboBox(parent)
    if combo_callback:
        combo_box.currentIndexChanged.connect(combo_callback)
    layout.addWidget(combo_box)

    return layout, combo_box, button


def create_horizontal_combo_with_button(
    button_text: str = "+",
    button_callback=None,
    combo_callback=None,
    parent=None,
) -> tuple:
    """Create a horizontal layout with a combo box and a button."""
    layout = QHBoxLayout()
    combo_box = CappedPopupComboBox(parent)
    if combo_callback:
        combo_box.currentIndexChanged.connect(combo_callback)

    button = QPushButton(button_text, parent)
    button.setMaximumWidth(30)
    if button_callback:
        button.clicked.connect(button_callback)

    layout.addWidget(combo_box, 1)
    layout.addWidget(button, 0)

    return layout, combo_box, button


def get_path_display_name(path):
    if not path:
        return ""
    if os.path.isfile(path) or path.endswith((".json", ".txt")):
        return os.path.basename(path)
    else:
        path = path.rstrip(os.path.sep)
        return os.path.basename(path)


def populate_dropdown(dropdown, recents, clear_first=True):
    if clear_first:
        dropdown.clear()
        dropdown.addItem("")

    for item in recents:
        if isinstance(item, dict) and "name" in item:
            display_text = item["name"]
            dropdown.addItem(display_text, userData=item)
        else:
            display_text = get_path_display_name(item)
            dropdown.addItem(display_text, userData=item)

    dropdown.setEditable(False)


class AtlasInstallationDialog(QDialog):
    """Dialog for installing or adding atlases."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install Atlas")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.radio_group = QButtonGroup(self)
        self.brain_globe_radio = QRadioButton("Install BrainGlobe Atlas")
        self.custom_radio = QRadioButton("Add Custom Atlas")
        self.radio_group.addButton(self.brain_globe_radio)
        self.radio_group.addButton(self.custom_radio)

        layout.addWidget(self.brain_globe_radio)
        layout.addWidget(self.custom_radio)

        # BrainGlobe Group
        self.brain_globe_group = QGroupBox("BrainGlobe Atlas")
        bg_layout = QVBoxLayout(self.brain_globe_group)
        self.brain_globe_combo = CappedPopupComboBox(self)
        self.install_button = QPushButton("Install")
        bg_layout.addWidget(self.brain_globe_combo)
        bg_layout.addWidget(self.install_button)
        layout.addWidget(self.brain_globe_group)

        # Custom Group
        self.custom_group = QGroupBox("Custom Atlas")
        custom_layout = QGridLayout(self.custom_group)

        self.name_edit = QLineEdit()
        self.path_edit = QLineEdit()
        self.label_edit = QLineEdit()
        self.browse_path_btn = QPushButton("Browse")
        self.browse_label_btn = QPushButton("Browse")
        self.add_custom_btn = QPushButton("Add Custom Atlas")

        custom_layout.addWidget(QLabel("Atlas Name:"), 0, 0)
        custom_layout.addWidget(self.name_edit, 0, 1, 1, 2)
        custom_layout.addWidget(QLabel("Atlas Path:"), 1, 0)
        custom_layout.addWidget(self.path_edit, 1, 1)
        custom_layout.addWidget(self.browse_path_btn, 1, 2)
        custom_layout.addWidget(QLabel("Label Path:"), 2, 0)
        custom_layout.addWidget(self.label_edit, 2, 1)
        custom_layout.addWidget(self.browse_label_btn, 2, 2)
        custom_layout.addWidget(self.add_custom_btn, 3, 0, 1, 3)
        layout.addWidget(self.custom_group)

        self.brain_globe_radio.toggled.connect(self.brain_globe_group.setVisible)
        self.custom_radio.toggled.connect(self.custom_group.setVisible)
        self.brain_globe_radio.setChecked(True)
        self.custom_group.setVisible(False)


def create_run_buttons_layout():
    """Create a horizontal layout with Run and Cancel buttons."""
    layout = QHBoxLayout()
    run_button = QPushButton("Run")
    cancel_button = QPushButton("Cancel")
    cancel_button.setEnabled(False)

    layout.addWidget(run_button)
    layout.addWidget(cancel_button)

    return layout, run_button, cancel_button


def select_path(
    parent,
    path_type="file",
    title="Select",
    update_function=None,
    key=None,
    dropdown=None,
    filter="",
):
    """Unified function for selecting files or directories."""
    if path_type == "file":
        path, _ = QFileDialog.getOpenFileName(parent, title, "", filter)
    else:
        path = QFileDialog.getExistingDirectory(parent, title)

    if path:
        if update_function and key:
            update_function(key, path)

        if dropdown and update_function:
            recent_files = parent.recent_files[key]
            populate_dropdown(dropdown, recent_files)
            dropdown.setCurrentIndex(1)

    return path


def create_path_selection_section(
    parent,
    label_text,
    path_type="file",
    title="Select",
    key=None,
    recents=None,
    callback=None,
) -> tuple:
    """Create a complete section for path selection (label, button, dropdown)."""
    def on_button_click():
        select_path(
            parent=parent,
            path_type=path_type,
            title=title,
            update_function=parent.update_recent,
            key=key,
            dropdown=dropdown,
        )

    layout, dropdown, button = create_labeled_combo_with_button(
        label_text,
        button_callback=on_button_click,
        parent=parent,
    )

    if recents:
        populate_dropdown(dropdown, recents)

    if callback:
        dropdown.currentIndexChanged.connect(callback)

    return layout, dropdown, button

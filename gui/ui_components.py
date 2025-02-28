"""
UI Component utilities for PyNutil GUI
"""

from PyQt6.QtWidgets import (
    QComboBox, QLabel, QHBoxLayout, QPushButton, QVBoxLayout,
    QFileDialog, QDialog, QRadioButton, QButtonGroup, QGroupBox,
    QLineEdit, QGridLayout
)
from PyQt6.QtCore import Qt
import os

def create_labeled_combo_with_button(
    label_text: str,
    button_text: str = "Browse...",
    button_callback=None,
    combo_callback=None,
    button_tooltip: str = None,
    button_width: int = None,
    parent=None,
    layout_type: str = "vertical",
    spacing: int = 6,  # Add spacing parameter (default Qt spacing is 6)
    margins: tuple = None  # Add margins parameter (left, top, right, bottom)
) -> tuple:
    """
    Create a labeled combo box with an associated button.

    Args:
        label_text: Text for the label
        button_text: Text for the button
        button_callback: Function to call when button is clicked
        combo_callback: Function to call when combo box selection changes
        button_tooltip: Tooltip for the button
        button_width: Width of the button
        parent: Parent widget
        layout_type: Type of layout - "vertical" (default), "horizontal", or "label_on_top"
        spacing: Spacing between widgets in the layout
        margins: Tuple of (left, top, right, bottom) margins, or None for default

    Returns:
        tuple: (layout, combo_box, button)
    """
    if layout_type == "horizontal":
        layout = QHBoxLayout()
        # Set spacing
        layout.setSpacing(spacing)

        # Set margins if provided
        if margins:
            layout.setContentsMargins(*margins)

        # Add widgets
        layout.addWidget(QLabel(label_text))

        # Create combo box
        combo_box = QComboBox(parent)
        combo_box.setStyleSheet("QComboBox { combobox-popup: 0; }")
        if combo_callback:
            combo_box.currentIndexChanged.connect(combo_callback)
        layout.addWidget(combo_box, 1)  # Give combo box stretch factor

        # Create button
        button = QPushButton(button_text, parent)
        if button_callback:
            button.clicked.connect(button_callback)
        if button_tooltip:
            button.setToolTip(button_tooltip)
        if button_width:
            button.setMaximumWidth(button_width)
        layout.addWidget(button)

    elif layout_type == "label_on_top":
        layout = QVBoxLayout()
        # Set spacing
        layout.setSpacing(spacing)

        # Set margins if provided
        if margins:
            layout.setContentsMargins(*margins)

        # Add label
        layout.addWidget(QLabel(label_text))

        # Create horizontal layout for combo box and button
        h_layout = QHBoxLayout()
        h_layout.setSpacing(spacing)

        # Create combo box
        combo_box = QComboBox(parent)
        combo_box.setStyleSheet("QComboBox { combobox-popup: 0; }")
        if combo_callback:
            combo_box.currentIndexChanged.connect(combo_callback)
        h_layout.addWidget(combo_box, 1)  # Give combo box stretch factor

        # Create button
        button = QPushButton(button_text, parent)
        if button_callback:
            button.clicked.connect(button_callback)
        if button_tooltip:
            button.setToolTip(button_tooltip)
        if button_width:
            button.setMaximumWidth(button_width)
        h_layout.addWidget(button)

        # Add horizontal layout to main layout
        layout.addLayout(h_layout)

    else:  # vertical (default)
        layout = QVBoxLayout()
        # Set spacing
        layout.setSpacing(spacing)

        # Set margins if provided
        if margins:
            layout.setContentsMargins(*margins)

        # Add label
        layout.addWidget(QLabel(label_text))

        # Create button
        button = QPushButton(button_text, parent)
        if button_callback:
            button.clicked.connect(button_callback)
        if button_tooltip:
            button.setToolTip(button_tooltip)
        if button_width:
            button.setMaximumWidth(button_width)

        # Add button
        layout.addWidget(button)

        # Create combo box
        combo_box = QComboBox(parent)
        combo_box.setStyleSheet("QComboBox { combobox-popup: 0; }")
        if combo_callback:
            combo_box.currentIndexChanged.connect(combo_callback)

        # Add combo box
        layout.addWidget(combo_box)

    return layout, combo_box, button

def create_horizontal_combo_with_button(
    combo_stretch: int = 1,
    button_text: str = "+",
    button_tooltip: str = None,
    button_width: int = 30,
    button_callback=None,
    combo_callback=None,
    parent=None,
    spacing: int = 1,  # Add spacing parameter
    margins: tuple = None  # Add margins parameter
) -> tuple:
    """
    Create a horizontal layout with a combo box and a button.

    Returns:
        tuple: (layout, combo_box, button)
    """
    layout = QHBoxLayout()
    # Set spacing
    layout.setSpacing(spacing)

    # Set margins if provided
    if margins:
        layout.setContentsMargins(*margins)

    # Create combo box
    combo_box = QComboBox(parent)
    combo_box.setStyleSheet("QComboBox { combobox-popup: 0; }")
    if combo_callback:
        combo_box.currentIndexChanged.connect(combo_callback)

    # Create button
    button = QPushButton(button_text, parent)
    if button_tooltip:
        button.setToolTip(button_tooltip)
    if button_width:
        button.setMaximumWidth(button_width)
    if button_callback:
        button.clicked.connect(button_callback)

    # Add widgets to layout
    layout.addWidget(combo_box, combo_stretch)  # Stretch factor for combo box
    layout.addWidget(button, 0)  # No stretch for button

    return layout, combo_box, button

def get_path_display_name(path):
    """Extract the filename or last directory from a path for display."""
    if not path:
        return ""
    if os.path.isfile(path) or path.endswith((".json", ".txt")):
        return os.path.basename(path)
    else:
        # For directories, show the last directory name
        path = path.rstrip(os.path.sep)  # Remove trailing slashes
        return os.path.basename(path)

def populate_dropdown(dropdown, recents, clear_first=True):
    """
    Populate a dropdown with items.

    Args:
        dropdown: QComboBox to populate
        recents: List of items to add (can be strings or user data)
        clear_first: Whether to clear the dropdown first (default: True)
    """
    if clear_first:
        dropdown.clear()
        dropdown.addItem("")

    # Store the full paths as user data but show shortened displays
    for item in recents:
        if isinstance(item, dict) and "name" in item:
            # Handle dictionary items (like custom atlases)
            display_text = item["name"]
            dropdown.addItem(display_text, userData=item)
        else:
            # Handle string items
            display_text = get_path_display_name(item)
            dropdown.addItem(display_text)
            # Store the full path as user data in the item
            dropdown.setItemData(dropdown.count() - 1, item)

    dropdown.setEditable(False)
    dropdown.setCurrentIndex(-1)

def create_atlas_installation_dialog(parent=None):
    """Create a dialog for installing or adding atlases."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Install Atlas")

    layout = QVBoxLayout(dialog)

    # Radio buttons for selecting atlas type
    radio_group = QButtonGroup(dialog)
    brain_globe_radio = QRadioButton("Install BrainGlobe Atlas")
    custom_radio = QRadioButton("Add Custom Atlas")
    radio_group.addButton(brain_globe_radio)
    radio_group.addButton(custom_radio)

    layout.addWidget(brain_globe_radio)
    layout.addWidget(custom_radio)

    # Group box for BrainGlobe atlas installation
    brain_globe_group = QGroupBox("BrainGlobe Atlas")
    brain_globe_layout = QVBoxLayout()
    brain_globe_group.setLayout(brain_globe_layout)

    brain_globe_combo = QComboBox()
    brain_globe_layout.addWidget(brain_globe_combo)

    install_brain_globe_button = QPushButton("Install")
    brain_globe_layout.addWidget(install_brain_globe_button)

    layout.addWidget(brain_globe_group)

    # Group box for custom atlas addition
    custom_group = QGroupBox("Custom Atlas")
    custom_layout = QGridLayout()
    custom_group.setLayout(custom_layout)

    custom_layout.addWidget(QLabel("Atlas Name:"), 0, 0)
    custom_atlas_name_edit = QLineEdit()
    custom_layout.addWidget(custom_atlas_name_edit, 0, 1)

    custom_layout.addWidget(QLabel("Atlas Path:"), 1, 0)
    custom_atlas_path_edit = QLineEdit()
    custom_layout.addWidget(custom_atlas_path_edit, 1, 1)
    browse_atlas_button = QPushButton("Browse")
    custom_layout.addWidget(browse_atlas_button, 1, 2)

    custom_layout.addWidget(QLabel("Label Path:"), 2, 0)
    custom_label_path_edit = QLineEdit()
    custom_layout.addWidget(custom_label_path_edit, 2, 1)
    browse_label_button = QPushButton("Browse")
    custom_layout.addWidget(browse_label_button, 2, 2)

    add_custom_button = QPushButton("Add Custom Atlas")
    custom_layout.addWidget(add_custom_button, 3, 0, 1, 3)

    layout.addWidget(custom_group)

    # Show/hide group boxes based on selected radio button
    brain_globe_radio.toggled.connect(brain_globe_group.setVisible)
    custom_radio.toggled.connect(custom_group.setVisible)

    brain_globe_radio.setChecked(True)
    custom_group.setVisible(False)

    dialog.setLayout(layout)

    return (
        dialog,
        brain_globe_radio, custom_radio,
        brain_globe_group, custom_group,
        brain_globe_combo,
        install_brain_globe_button,
        custom_atlas_name_edit, custom_atlas_path_edit, custom_label_path_edit,
        browse_atlas_button, browse_label_button, add_custom_button
    )

def create_run_buttons_layout(spacing: int = 0, margins: tuple = None):
    """Create a horizontal layout with Run and Cancel buttons."""
    layout = QHBoxLayout()

    # Set spacing
    layout.setSpacing(spacing)

    # Set margins if provided
    if margins:
        layout.setContentsMargins(*margins)

    run_button = QPushButton("Run")
    cancel_button = QPushButton("Cancel")
    cancel_button.setEnabled(False)  # Disabled by default

    layout.addWidget(run_button)
    layout.addWidget(cancel_button)

    return layout, run_button, cancel_button

def select_path(parent,
               path_type="file",
               title="Select",
               update_function=None,
               key=None,
               dropdown=None,
               argument_dict=None,
               filter=""):
    """
    Unified function for selecting files or directories.

    Args:
        parent: Parent widget
        path_type: "file" or "directory"
        title: Dialog title
        update_function: Function to update recent files list
        key: Key for recent files dictionary
        dropdown: Dropdown to update
        argument_dict: Dictionary to update with path
        filter: File filter (only used for files)

    Returns:
        Selected path
    """
    if path_type == "file":
        path, _ = QFileDialog.getOpenFileName(parent, title, "", filter)
    else:
        path = QFileDialog.getExistingDirectory(parent, title)

    if path:
        # Update dictionary if provided
        if argument_dict is not None and key is not None:
            argument_dict[key] = path

        # Update recent files if function provided
        if update_function and key:
            update_function(key, path)

        # Update dropdown if provided
        if dropdown and update_function:
            # Get updated recents list
            recent_files = parent.recent_files[key]
            populate_dropdown(dropdown, recent_files)
            dropdown.setCurrentIndex(1)  # Select first item (newest)

    return path

def create_path_selection_section(parent,
                                 label_text,
                                 path_type="file",
                                 button_text="Browse...",
                                 title="Select",
                                 key=None,
                                 filter="",
                                 recents=None,
                                 callback=None,
                                 argument_dict=None,
                                 layout_type="label_on_top",
                                 spacing: int = 3,  # Add spacing with a smaller default
                                 margins: tuple = (0, 0, 0, 0)):  # Add margins with zero default
    """
    Create a complete section for path selection (label, button, dropdown).

    Args:
        parent: Parent widget
        label_text: Label text
        path_type: "file" or "directory"
        button_text: Button text
        title: Dialog title
        key: Key for recent files dictionary
        filter: File filter (only used for files)
        recents: Recent paths list
        callback: Function to be called when dropdown selection changes
        argument_dict: Dictionary to update with path
        layout_type: Type of layout for components
        spacing: Spacing between widgets
        margins: Tuple of (left, top, right, bottom) margins

    Returns:
        tuple: (layout, dropdown, button)
    """
    def on_button_click():
        select_path(
            parent=parent,
            path_type=path_type,
            title=title,
            update_function=parent.update_recent,
            key=key,
            dropdown=dropdown,
            argument_dict=argument_dict
        )

    # Create layout with label, button and dropdown
    layout, dropdown, button = create_labeled_combo_with_button(
        label_text,
        button_text=button_text,
        button_callback=on_button_click,
        layout_type=layout_type,
        spacing=spacing,
        margins=margins
    )

    # Populate dropdown with recent items if provided
    if recents:
        populate_dropdown(dropdown, recents)

    # Connect dropdown selection change to callback if provided
    if callback:
        dropdown.currentIndexChanged.connect(callback)

    return layout, dropdown, button

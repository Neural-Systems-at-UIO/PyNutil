import json
import os
from typing import Dict, List, Any, Optional


class SettingsManager:
    """Manages application settings and recent files."""

    def __init__(self, settings_path: str):
        """
        Initialize the settings manager.

        Args:
            settings_path: Path to the settings file
        """
        self.settings_path = settings_path
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file."""
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r") as file:
                    data = json.load(file)
                    # Ensure all required keys exist
                    for key in ["registration_json", "segmentation_dir", "output_dir"]:
                        if not isinstance(data.get(key, []), list):
                            data[key] = [data.get(key)] if data.get(key) else []
                    if "object_colour" not in data:
                        data["object_colour"] = []
                    if "custom_atlases" not in data:
                        data["custom_atlases"] = []
                    return data
            except Exception as e:
                print(f"Error loading settings: {e}")

        # Return default settings if file doesn't exist or has errors
        return {
            "registration_json": [],
            "segmentation_dir": [],
            "output_dir": [],
            "object_colour": [],
            "custom_atlases": [],
        }

    def save_settings(self) -> None:
        """Save settings to file."""
        try:
            with open(self.settings_path, "w") as file:
                json.dump(self.settings, file)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def update_recent(self, key: str, value: str) -> None:
        """
        Update recent files list for a specific setting.

        Args:
            key: Setting key
            value: New value to add to recent list
        """
        recents = self.settings.get(key, [])
        if value in recents:
            recents.remove(value)
        recents.insert(0, value)
        self.settings[key] = recents[:5]  # Keep only 5 most recent
        self.save_settings()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ValidationResult:
    missing_fields: List[str]

    @property
    def ok(self) -> bool:
        return len(self.missing_fields) == 0

    def to_html(self) -> str:
        if self.ok:
            return ""
        lines = ["Error: The following required settings are missing:<br>"]
        for field in self.missing_fields:
            lines.append(f"- {field}<br>")
        lines.append("<br>Please provide all required settings before running the analysis.")
        return "".join(lines)


def validate_analysis_inputs(
    *,
    atlas_text: str,
    arguments: Dict[str, Any],
    custom_atlas_data: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    missing: List[str] = []

    if not atlas_text:
        missing.append("Reference Atlas")

    if not arguments.get("registration_json"):
        missing.append("Registration JSON")

    if not arguments.get("segmentation_dir"):
        missing.append("Segmentation Folder")

    colour = arguments.get("object_colour")
    if not colour:
        missing.append("Object Color")

    if not arguments.get("output_dir"):
        missing.append("Output Directory")

    # If user selected a custom atlas entry, ensure its paths exist.
    if custom_atlas_data:
        if not custom_atlas_data.get("atlas_path"):
            missing.append("Custom Atlas Path")
        if not custom_atlas_data.get("label_path"):
            missing.append("Custom Label Path")

    return ValidationResult(missing_fields=missing)

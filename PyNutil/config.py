from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _unwrap_singleton_list(value: Any) -> Any:
    """Unwrap legacy settings fields like [null] -> None, ["x"] -> "x"."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


@dataclass
class PyNutilConfig:
    segmentation_folder: Optional[str] = None
    image_folder: Optional[str] = None
    alignment_json: Optional[str] = None
    colour: Optional[list] = None
    intensity_channel: Optional[str] = None
    atlas_name: Optional[str] = None
    atlas_path: Optional[str] = None
    label_path: Optional[str] = None
    hemi_path: Optional[str] = None
    custom_region_path: Optional[str] = None
    voxel_size_um: Optional[float] = None
    min_intensity: Optional[int] = None
    max_intensity: Optional[int] = None
    cellpose: bool = False

    @classmethod
    def from_settings_file(cls, settings_file: str) -> "PyNutilConfig":
        with open(settings_file, "r") as f:
            settings = json.load(f)
        return cls.from_settings_dict(settings)

    @classmethod
    def from_settings_dict(cls, settings: Dict[str, Any]) -> "PyNutilConfig":
        # Be tolerant of legacy settings saved as singleton lists.
        def g(key: str, default: Any = None) -> Any:
            return _unwrap_singleton_list(settings.get(key, default))

        # alignment_json is required in settings files.
        if "alignment_json" not in settings:
            raise KeyError(
                "Settings file must contain alignment_json, and either atlas_path and label_path or atlas_name. It should also contain either segmentation_folder or image_folder."
            )

        cfg = cls(
            segmentation_folder=g("segmentation_folder"),
            image_folder=g("image_folder"),
            alignment_json=settings["alignment_json"],
            colour=g("colour"),
            intensity_channel=g("intensity_channel"),
            atlas_name=g("atlas_name"),
            atlas_path=g("atlas_path"),
            label_path=g("label_path"),
            hemi_path=g("hemi_path"),
            custom_region_path=g("custom_region_path"),
            voxel_size_um=g("voxel_size_um"),
            min_intensity=g("min_intensity"),
            max_intensity=g("max_intensity"),
            cellpose=bool(g("cellpose", False)),
        )

        # If atlas_path/label_path are present but empty/null, treat as not provided.
        if not cfg.atlas_path or not cfg.label_path:
            cfg.atlas_path = None
            cfg.label_path = None
            cfg.hemi_path = None

        return cfg

    def normalize(self, *, logger=None) -> "PyNutilConfig":
        # If atlas_name is provided, voxel size is inferred from atlas name and
        # any manually provided voxel_size_um is ignored (existing behavior).
        if self.atlas_name is not None and self.voxel_size_um is not None:
            if logger is not None:
                logger.warning(
                    f"Voxel size ({self.voxel_size_um} um) was specified but will be ignored because atlas_name ({self.atlas_name}) is provided. Voxel size will be inferred from the atlas name."
                )
            self.voxel_size_um = None
        return self

    def validate(self) -> None:
        if self.segmentation_folder and self.image_folder:
            raise ValueError(
                "Please specify either segmentation_folder or image_folder, not both."
            )

        if self.segmentation_folder and (
            self.min_intensity is not None or self.max_intensity is not None
        ):
            raise ValueError(
                "min_intensity and max_intensity are only supported when using image_folder, not segmentation_folder."
            )

        if self.image_folder and (self.colour is not None):
            raise ValueError(
                "You can't specify both colour and image_folder since there are no segmentations"
            )

        if (self.atlas_path or self.label_path) and self.atlas_name:
            raise ValueError(
                "Please specify either atlas_path and label_path or atlas_name. Atlas and label paths are only used for loading custom atlases."
            )

        if not (self.atlas_path and self.label_path) and not self.atlas_name:
            raise ValueError(
                "When atlas_path and label_path are not specified, atlas_name must be specified."
            )

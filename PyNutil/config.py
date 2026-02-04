from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


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
    segmentation_format: str = "binary"  # "binary" or "cellpose"

    @classmethod
    def from_settings_file(cls, settings_file: str) -> "PyNutilConfig":
        with open(settings_file, "r") as f:
            settings = json.load(f)
        return cls.from_settings_dict(settings)

    @classmethod
    def from_settings_dict(cls, settings: Dict[str, Any]) -> "PyNutilConfig":
        # alignment_json is required in settings files.
        if "alignment_json" not in settings:
            raise KeyError(
                "Settings file must contain alignment_json, and either atlas_path and label_path or atlas_name. It should also contain either segmentation_folder or image_folder."
            )

        cfg = cls(
            segmentation_folder=settings.get("segmentation_folder"),
            image_folder=settings.get("image_folder"),
            alignment_json=settings["alignment_json"],
            colour=settings.get("colour"),
            intensity_channel=settings.get("intensity_channel"),
            atlas_name=settings.get("atlas_name"),
            atlas_path=settings.get("atlas_path"),
            label_path=settings.get("label_path"),
            hemi_path=settings.get("hemi_path"),
            custom_region_path=settings.get("custom_region_path"),
            voxel_size_um=settings.get("voxel_size_um"),
            min_intensity=settings.get("min_intensity"),
            max_intensity=settings.get("max_intensity"),
            segmentation_format=settings.get("segmentation_format", "binary"),
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

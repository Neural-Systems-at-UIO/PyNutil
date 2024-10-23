import json
import os
from datetime import datetime

import brainglobe_atlasapi
import numpy as np
import pandas as pd

from .coordinate_extraction import folder_to_atlas_space
from .counting_and_load import label_points, pixel_count_per_region
from .read_and_write import read_atlas_volume, write_points_to_meshview

class PyNutil:
    def __init__(self, segmentation_folder=None, alignment_json=None, colour=None, 
                 atlas_name=None, atlas_path=None, label_path=None, settings_file=None):
        if settings_file is not None:
            with open(settings_file, "r") as f:
                settings = json.load(f)
            try:
                segmentation_folder = settings["segmentation_folder"]
                alignment_json = settings["alignment_json"]
                colour = settings["colour"]
                atlas_name = settings["atlas_name"]
            except KeyError as exc:
                raise KeyError(
                    "settings file must contain segmentation_folder, alignment_json, colour, and atlas_name"
                ) from exc

        self.segmentation_folder = segmentation_folder
        self.alignment_json = alignment_json
        self.colour = colour
        self.atlas_name = atlas_name

        if (atlas_path or label_path) and atlas_name:
            raise ValueError(
                "Please only specify an atlas_path and a label_path or an atlas_name, atlas and label paths are only used for loading custom atlases"
            )

        if atlas_path and label_path:
            self.atlas_volume, self.atlas_labels = self.load_custom_atlas(atlas_path, label_path)
        else:
            self._check_atlas_name()
            self.atlas_volume, self.atlas_labels = self.load_atlas_data(atlas_name=atlas_name)

    def _check_atlas_name(self):
        if not self.atlas_name:
            raise ValueError("Atlas name must be specified")

    def _load_settings(self, settings_file):
        if settings_file:
            with open(settings_file, "r") as f:
                settings = json.load(f)
            self.segmentation_folder = settings.get("segmentation_folder")
            self.alignment_json = settings.get("alignment_json")
            self.colour = settings.get("colour")
            self.atlas_name = settings.get("atlas_name")


    def load_atlas_data(self, atlas_name):
        """Loads the atlas volume and labels from disk."""
        atlas = brainglobe_atlasapi.BrainGlobeAtlas(atlas_name=atlas_name)
        atlas_structures = {
            "idx": [i["id"] for i in atlas.structures_list],
            "name": [i["name"] for i in atlas.structures_list],
            "r": [i["rgb_triplet"][0] for i in atlas.structures_list],
            "g": [i["rgb_triplet"][1] for i in atlas.structures_list],
            "b": [i["rgb_triplet"][2] for i in atlas.structures_list],
        }
        atlas_structures["idx"].insert(0, 0)
        atlas_structures["name"].insert(0, "Clear Label")
        atlas_structures["r"].insert(0, 0)
        atlas_structures["g"].insert(0, 0)
        atlas_structures["b"].insert(0, 0)

        atlas_labels = pd.DataFrame(atlas_structures)
        atlas_volume = self._process_atlas_volume(atlas)
        print("atlas labels loaded ✅")
        return atlas_volume, atlas_labels

    def _process_atlas_volume(self, atlas):
        print("reorienting brainglobe atlas into quicknii space...")
        return np.transpose(atlas.annotation, [2, 0, 1])[:, ::-1, ::-1]
 

    def load_custom_atlas(self, atlas_path, label_path):
        atlas_volume = read_atlas_volume(atlas_path)
        atlas_labels = pd.read_csv(label_path)
        return atlas_volume, atlas_labels

    def get_coordinates(self, non_linear=True, object_cutoff=0, use_flat=False):
        """Extracts pixel coordinates from the segmentation data."""
        pixel_points, centroids, region_areas_list, points_len, centroids_len, segmentation_filenames = folder_to_atlas_space(
            self.segmentation_folder,
            self.alignment_json,
            self.atlas_labels,
            pixel_id=self.colour,
            non_linear=non_linear,
            object_cutoff=object_cutoff,
            atlas_volume=self.atlas_volume,
            use_flat=use_flat,
        )
        self.pixel_points = pixel_points
        self.centroids = centroids
        self.points_len = points_len
        self.centroids_len = centroids_len
        self.segmentation_filenames = segmentation_filenames
        self.region_areas_list = region_areas_list


    def quantify_coordinates(self):
        """Quantifies the pixel coordinates by region."""
        self._check_coordinates_extracted()
        print("quantifying coordinates")
        labeled_points_centroids = self._label_points(self.centroids)
        labeled_points = self._label_points(self.pixel_points) 

        self._quantify_per_section(labeled_points, labeled_points_centroids)
        self._combine_slice_reports()

        self.labeled_points = labeled_points
        self.labeled_points_centroids = labeled_points_centroids

        print("quantification complete ✅")

    def _check_coordinates_extracted(self):
        if not hasattr(self, "pixel_points") and not hasattr(self, "centroids"):
            raise ValueError("Please run get_coordinates before running quantify_coordinates")

    def _label_points(self, points):
        return label_points(points, self.atlas_volume, scale_factor=1)

    def _quantify_per_section(self, labeled_points, labeled_points_centroids):
        prev_pl = 0
        prev_cl = 0
        per_section_df = []

        for pl, cl, ra in zip(self.points_len, self.centroids_len, self.region_areas_list):
            current_centroids = labeled_points_centroids[prev_cl : prev_cl + cl] 
            current_points = labeled_points[prev_pl : prev_pl + pl] 
            current_df = pixel_count_per_region(current_points, current_centroids, self.atlas_labels)
            current_df_new = self._merge_dataframes(current_df, ra)
            per_section_df.append(current_df_new)
            prev_pl += pl
            prev_cl += cl

        self.per_section_df = per_section_df

    def _merge_dataframes(self, current_df, ra):
        all_region_df = self.atlas_labels.merge(ra, on="idx", how="left")
        current_df_new = all_region_df.merge(current_df, on="idx", how="left", suffixes=(None, "_y")).drop(columns=["name_y", "r_y", "g_y", "b_y"])
        current_df_new["area_fraction"] = current_df_new["pixel_count"] / current_df_new["region_area"]
        current_df_new.fillna(0, inplace=True)
        return current_df_new

    def _combine_slice_reports(self):
        self.label_df = (
            pd.concat(self.per_section_df)
            .groupby(["idx", "name", "r", "g", "b"])
            .sum()
            .reset_index()
            .drop(columns=["area_fraction"])
        )
        self.label_df["area_fraction"] = self.label_df["pixel_count"] / self.label_df["region_area"]
        self.label_df.fillna(0, inplace=True)

        self.label_df = self.label_df.set_index("idx")
        self.label_df = self.label_df.reindex(index=self.atlas_labels["idx"])
        self.label_df = self.label_df.reset_index()

    def save_analysis(self, output_folder):
        """Saves the pixel coordinates and pixel counts to different files in the specified
        output folder."""
        self._create_output_dirs(output_folder)
        self._save_quantification(output_folder)
        self._save_per_section_reports(output_folder)
        self._save_whole_series_meshview(output_folder)
        print("analysis saved ✅")

    def _create_output_dirs(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(f"{output_folder}/whole_series_report", exist_ok=True)
        os.makedirs(f"{output_folder}/per_section_meshview", exist_ok=True)
        os.makedirs(f"{output_folder}/per_section_reports", exist_ok=True)
        os.makedirs(f"{output_folder}/whole_series_meshview", exist_ok=True)

    def _save_quantification(self, output_folder):
        if hasattr(self, "label_df"):
            self.label_df.to_csv(f"{output_folder}/whole_series_report/counts.csv", sep=";", na_rep="", index=False)
        else:
            print("no quantification found so we will only save the coordinates")
            print("if you want to save the quantification please run quantify_coordinates")

    def _save_per_section_reports(self, output_folder):
        prev_pl = 0
        prev_cl = 0

        for pl, cl, fn, df in zip(self.points_len, self.centroids_len, self.segmentation_filenames, self.per_section_df):
            split_fn = fn.split(os.sep)[-1].split(".")[0]
            df.to_csv(f"{output_folder}/per_section_reports/{split_fn}.csv", sep=";", na_rep="", index=False)
            self._save_per_section_meshview(output_folder, split_fn, pl, cl, prev_pl, prev_cl)
            prev_cl += cl
            prev_pl += pl

    def _save_per_section_meshview(self, output_folder, split_fn, pl, cl, prev_pl, prev_cl):
            write_points_to_meshview(self.pixel_points[prev_pl : pl + prev_pl], self.labeled_points[prev_pl : pl + prev_pl], f"{output_folder}/per_section_meshview/{split_fn}_pixels.json", self.atlas_labels)
            write_points_to_meshview(self.centroids[prev_cl : cl + prev_cl], self.labeled_points_centroids[prev_cl : cl + prev_cl], f"{output_folder}/per_section_meshview/{split_fn}_centroids.json", self.atlas_labels)

    def _save_whole_series_meshview(self, output_folder):
            write_points_to_meshview(self.pixel_points, self.labeled_points, f"{output_folder}/whole_series_meshview/pixels_meshview.json", self.atlas_labels)
            write_points_to_meshview(self.centroids, self.labeled_points_centroids, f"{output_folder}/whole_series_meshview/objects_meshview.json", self.atlas_labels)
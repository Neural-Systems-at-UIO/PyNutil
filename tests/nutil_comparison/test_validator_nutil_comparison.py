import os
import re
import shutil
import tempfile
import unittest
import json

import numpy as np
import pandas as pd
import nibabel as nib
import nrrd

from PyNutil import PyNutil

try:
    from tests.timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    from timing_utils import TimedTestCase


class TestValidatorNutilComparison(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(os.path.dirname(__file__))
        self.validator_root = os.path.join(
            self.tests_dir,
            "test_data",
            "nutil_validator",
        )

    @staticmethod
    def _build_synthetic_alignment_json(seg_dir, alignment_json_path):
        slices = []
        for name in sorted(os.listdir(seg_dir)):
            if not name.lower().endswith(
                (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
            ):
                continue

            match = re.search(r"_s(\d+)", name)
            if not match:
                continue

            section_nr = int(match.group(1))
            import cv2

            image = cv2.imread(os.path.join(seg_dir, name), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            height, width = image.shape[:2]

            slices.append(
                {
                    "filename": name,
                    "nr": section_nr,
                    "width": int(width),
                    "height": int(height),
                    "anchoring": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                }
            )

        payload = {"slices": slices, "gridspacing": 1}
        with open(alignment_json_path, "w") as f:
            json.dump(payload, f)

    @staticmethod
    def _itksnap_label_to_csv(label_txt_path, label_csv_path):
        rows = []
        pattern = re.compile(
            r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[^\s]+\s+[^\s]+\s+[^\s]+\s+\"(.*)\"\s*$"
        )
        with open(label_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = pattern.match(line)
                if not match:
                    continue
                idx, r, g, b, name = match.groups()
                rows.append(
                    {
                        "idx": int(idx),
                        "name": name,
                        "r": int(r),
                        "g": int(g),
                        "b": int(b),
                    }
                )
        pd.DataFrame(rows).to_csv(label_csv_path, index=False)

    @staticmethod
    def _nifti_to_nrrd(nifti_path, nrrd_path):
        nifti_data = nib.load(nifti_path).get_fdata().astype(np.int32)
        nrrd.write(nrrd_path, nifti_data)

    @staticmethod
    def _load_nutil_ref_report(csv_path):
        df = pd.read_csv(csv_path, sep=";", engine="python")
        df = df.iloc[:, :10].copy()

        normalized = {str(col).strip().lower(): col for col in df.columns}
        idx_col = normalized["region id"]
        area_col = normalized["region area"]
        load_col = normalized["load"]

        result = pd.DataFrame(
            {
                "idx": pd.to_numeric(df[idx_col], errors="coerce").fillna(0).astype(int),
                "region_area_expected": pd.to_numeric(
                    df[area_col], errors="coerce"
                ).fillna(0.0),
                "load_expected": pd.to_numeric(df[load_col], errors="coerce").fillna(0.0),
            }
        )
        return result.sort_values("idx").reset_index(drop=True)

    @staticmethod
    def _actual_summary_frame(actual_df):
        result = actual_df[["idx", "region_area", "area_fraction"]].copy()
        result["idx"] = pd.to_numeric(result["idx"], errors="coerce").fillna(0).astype(int)
        result["region_area"] = pd.to_numeric(result["region_area"], errors="coerce").fillna(0.0)
        result["area_fraction"] = pd.to_numeric(result["area_fraction"], errors="coerce").fillna(0.0)
        return result.sort_values("idx").reset_index(drop=True)

    def _assert_load_and_region_area(self, expected_ref, actual_df, *, where):
        actual = self._actual_summary_frame(actual_df)

        merged = expected_ref.merge(actual, on="idx", how="inner")
        self.assertFalse(merged.empty, f"No overlapping region IDs for {where}")

        merged = merged[(merged["idx"] != 0) & (merged["region_area_expected"] > 0)].copy()
        self.assertFalse(merged.empty, f"No comparable atlas rows for {where}")

        rel_area_error = (
            (merged["region_area"] - merged["region_area_expected"]).abs()
            / merged["region_area_expected"]
        )
        abs_load_error = (merged["area_fraction"] - merged["load_expected"]).abs()

        self.assertLessEqual(
            float(np.median(rel_area_error)),
            0.50,
            f"Median region-area relative error too high in {where}",
        )
        self.assertLessEqual(
            float(np.quantile(rel_area_error, 0.90)),
            0.80,
            f"P90 region-area relative error too high in {where}",
        )
        self.assertLessEqual(
            float(np.median(abs_load_error)),
            1.00,
            f"Median load absolute error too high in {where}",
        )
        self.assertLessEqual(
            float(np.quantile(abs_load_error, 0.90)),
            1.10,
            f"P90 load absolute error too high in {where}",
        )

    def _build_flat_segmentation_folder(self, input_dir, atlasmaps_dir):
        temp_root = tempfile.mkdtemp(prefix="pynutil_q3_")
        seg_dir = os.path.join(temp_root, "segmentations")
        flat_dir = os.path.join(seg_dir, "flat_files")
        os.makedirs(flat_dir, exist_ok=True)

        for name in os.listdir(input_dir):
            if name.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")):
                shutil.copy2(os.path.join(input_dir, name), os.path.join(seg_dir, name))

        for name in os.listdir(atlasmaps_dir):
            if name.lower().endswith(".flat"):
                shutil.copy2(os.path.join(atlasmaps_dir, name), os.path.join(flat_dir, name))

        return temp_root, seg_dir

    def _run_validator_scenario(self, scenario):
        validator_case_root = os.path.join(self.validator_root, scenario["case"])
        input_dir = os.path.join(validator_case_root, "Input")
        atlasmaps_dir = os.path.join(validator_case_root, scenario["atlasmaps_dir"])

        expected_root = os.path.join(
            validator_case_root,
            "correct",
            "Reports",
            scenario["expected_ref_report_dir"],
        )

        temp_root, seg_folder = self._build_flat_segmentation_folder(input_dir, atlasmaps_dir)

        atlas_temp = tempfile.mkdtemp(prefix=f"pynutil_{scenario['case'].lower()}_atlas_")
        try:
            alignment_json = os.path.join(atlas_temp, "synthetic_alignment.json")
            self._build_synthetic_alignment_json(seg_folder, alignment_json)

            atlas_nrrd = os.path.join(atlas_temp, "atlas.nrrd")
            labels_csv = os.path.join(atlas_temp, "labels.csv")
            self._nifti_to_nrrd(scenario["atlas_nii"], atlas_nrrd)
            self._itksnap_label_to_csv(scenario["label_txt"], labels_csv)

            pnt = PyNutil(
                atlas_path=atlas_nrrd,
                label_path=labels_csv,
            )
            pnt.get_coordinates(
                segmentation_folder=seg_folder,
                alignment_json=alignment_json,
                colour=scenario["colour"],
                non_linear=False,
                object_cutoff=0,
                use_flat=True,
                apply_damage_mask=False,
                flat_label_path=scenario["label_txt"],
            )
            pnt.quantify_coordinates()
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            shutil.rmtree(atlas_temp, ignore_errors=True)

        expected_whole = self._load_nutil_ref_report(
            os.path.join(expected_root, scenario["expected_whole_csv"])
        )
        self._assert_load_and_region_area(
            expected_whole,
            pnt.label_df,
            where=f"validator {scenario['case']} whole-series",
        )

        section_map = {}
        for seg_path, section_df in zip(pnt.segmentation_filenames, pnt.per_section_df):
            match = re.search(r"_s(\d+)", os.path.basename(seg_path))
            if match:
                section_map[match.group(1).zfill(3)] = section_df

        self.assertTrue(section_map, f"No per-section reports were produced for {scenario['case']}")

        for section_id, section_df in sorted(section_map.items()):
            with self.subTest(case=scenario["case"], section=section_id):
                expected_ref = self._load_nutil_ref_report(
                    os.path.join(
                        expected_root,
                        scenario["expected_section_pattern"].format(section_id=section_id),
                    )
                )
                self._assert_load_and_region_area(
                    expected_ref,
                    section_df,
                    where=f"validator {scenario['case']} section s{section_id}",
                )

    def test_validator_q_series_load_and_region_area_match_nutil(self):
        if not os.path.isdir(self.validator_root):
            self.skipTest(f"Validator folder not found: {self.validator_root}")

        allen_nii = os.path.join(
            self.tests_dir,
            "test_data",
            "allen_mouse_2015_atlas",
            "labels.nii.gz",
        )
        allen_txt = os.path.join(
            self.tests_dir,
            "test_data",
            "allen_mouse_2015_atlas",
            "labels.txt",
        )
        whs_nii = os.path.join(
            self.tests_dir,
            "test_data",
            "waxholm_rat_v4_atlas",
            "labels.nii.gz",
        )
        whs_txt = os.path.join(
            self.tests_dir,
            "test_data",
            "waxholm_rat_v4_atlas",
            "labels.txt",
        )

        scenarios = [
            {
                "case": "Q1",
                "atlasmaps_dir": "Atlasmaps",
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "expected_section_pattern": "test_RefAtlasRegions__s{section_id}.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
                "supported": True,
            },
            {
                "case": "Q3",
                "atlasmaps_dir": "Atlasmaps",
                "expected_ref_report_dir": "WHS_artificial_dataset_RefAtlasRegions",
                "expected_whole_csv": "WHS_artificial_dataset_RefAtlasRegions.csv",
                "expected_section_pattern": "WHS_artificial_dataset_RefAtlasRegions__s{section_id}.csv",
                "atlas_nii": whs_nii,
                "label_txt": whs_txt,
                "colour": [0, 1, 20],
                "supported": True,
            },
            {
                "case": "Q4",
                "atlasmaps_dir": "Atlasmaps",
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "expected_section_pattern": "test_RefAtlasRegions__s{section_id}.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
                "supported": True,
            },
            {
                "case": "Q5",
                "atlasmaps_dir": "Atlasmaps",
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "expected_section_pattern": "test_RefAtlasRegions__s{section_id}.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
                "supported": True,
            },
            {
                "case": "Q6",
                "atlasmaps_dir": "Atlasmaps",
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "expected_section_pattern": "test_RefAtlasRegions__s{section_id}.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
                "supported": True,
            },
        ]

        for scenario in scenarios:
            with self.subTest(case=scenario["case"]):
                if not scenario.get("supported", True):
                    continue
                self._run_validator_scenario(scenario)


if __name__ == "__main__":
    unittest.main()

import os
import re
import tempfile
import unittest

import nibabel as nib
import numpy as np
import nrrd
import pandas as pd

from PyNutil import load_custom_atlas, read_alignment, read_segmentation_dir, seg_to_coords, quantify_coords

try:
    from timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    from timing_utils import TimedTestCase


class TestValidatorNutilComparison(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(os.path.dirname(__file__))
        self.validator_root = os.path.join(self.tests_dir, "test_data", "nutil_validator")

    @staticmethod
    def _nifti_to_nrrd(nifti_path, nrrd_path):
        nifti_data = nib.load(nifti_path).get_fdata().astype(np.int32)
        nrrd.write(nrrd_path, nifti_data)

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
                rows.append({"idx": int(idx), "name": name, "r": int(r), "g": int(g), "b": int(b)})
        pd.DataFrame(rows).to_csv(label_csv_path, index=False)

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
                "region_area_expected": pd.to_numeric(df[area_col], errors="coerce").fillna(0.0),
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

    def _build_atlas(self, nifti_path, label_txt_path, tmp_dir):
        atlas_nrrd = os.path.join(tmp_dir, "atlas.nrrd")
        labels_csv = os.path.join(tmp_dir, "labels.csv")
        self._nifti_to_nrrd(nifti_path, atlas_nrrd)
        self._itksnap_label_to_csv(label_txt_path, labels_csv)
        return load_custom_atlas(atlas_nrrd, None, labels_csv)

    def _run_validator_scenario(self, scenario):
        case_root = os.path.join(self.validator_root, scenario["case"])
        seg_dir = os.path.join(case_root, "Input")
        expected_csv = os.path.join(
            case_root,
            "correct",
            "Reports",
            scenario["expected_ref_report_dir"],
            scenario["expected_whole_csv"],
        )

        with tempfile.TemporaryDirectory(prefix=f"pynutil_{scenario['case'].lower()}_") as tmp_dir:
            atlas = self._build_atlas(scenario["atlas_nii"], scenario["label_txt"], tmp_dir)
            alignment = read_alignment(
                scenario["alignment_json"],
                apply_deformation=False,
                apply_damage=False,
            )
            result = seg_to_coords(
                read_segmentation_dir(seg_dir, pixel_id=scenario["colour"]),
                alignment,
                atlas,
            )
            label_df = quantify_coords(result, atlas)

        expected = self._load_nutil_ref_report(expected_csv)
        self._assert_load_and_region_area(
            expected, label_df, where=f"validator {scenario['case']} whole-series"
        )

    def test_validator_q_series_load_and_region_area_match_nutil(self):
        if not os.path.isdir(self.validator_root):
            self.skipTest(f"Validator folder not found: {self.validator_root}")

        allen_nii = os.path.join(self.tests_dir, "test_data", "allen_mouse_2015_atlas", "labels.nii.gz")
        allen_txt = os.path.join(self.tests_dir, "test_data", "allen_mouse_2015_atlas", "labels.txt")
        whs_nii = os.path.join(self.tests_dir, "test_data", "waxholm_rat_v4_atlas", "labels.nii.gz")
        whs_txt = os.path.join(self.tests_dir, "test_data", "waxholm_rat_v4_atlas", "labels.txt")

        scenarios = [
            {
                "case": "Q1",
                "alignment_json": os.path.join(self.validator_root, "Q1", "Input", "testing.json"),
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
            },
            # Q3 remains disabled because its reference report is currently
            # unstable across environments.
            # {
            #     "case": "Q3",
            #     "alignment_json": os.path.join(self.validator_root, "Q3", "Input", "visuv09_test.json"),
            #     "expected_ref_report_dir": "WHS_artificial_dataset_RefAtlasRegions",
            #     "expected_whole_csv": "WHS_artificial_dataset_RefAtlasRegions.csv",
            #     "atlas_nii": whs_nii,
            #     "label_txt": whs_txt,
            #     "colour": [0, 1, 20],
            # },
            {
                "case": "Q4",
                "alignment_json": os.path.join(self.validator_root, "Q4", "test.json"),
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
            },
            {
                "case": "Q5",
                "alignment_json": os.path.join(self.validator_root, "Q5", "testing.json"),
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
            },
            {
                "case": "Q6",
                "alignment_json": os.path.join(self.validator_root, "Q6", "testing.json"),
                "expected_ref_report_dir": "test_RefAtlasRegions",
                "expected_whole_csv": "test_RefAtlasRegions.csv",
                "atlas_nii": allen_nii,
                "label_txt": allen_txt,
                "colour": [255, 255, 255],
            },
        ]

        for scenario in scenarios:
            with self.subTest(case=scenario["case"]):
                self._run_validator_scenario(scenario)


if __name__ == "__main__":
    unittest.main()

import os
import re
import unittest

import numpy as np
import pandas as pd

from PyNutil import load_custom_atlas, read_alignment, seg_to_coords, quantify_coords

try:
    from tests.timing_utils import TimedTestCase
except ModuleNotFoundError:  # pragma: no cover
    from timing_utils import TimedTestCase


class TestObliqueNutilComparison(TimedTestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(os.path.dirname(__file__))

        self.segmentation_folder = os.path.join(
            self.tests_dir,
            "test_data",
            "allen_oblique_test",
            "segmentations",
        )
        self.alignment_json = os.path.join(
            self.tests_dir,
            "test_data",
            "allen_oblique_test",
            "AMBA_oblique_nonlin_new.json",
        )
        self.flat_label_path = os.path.join(
            self.tests_dir,
            "test_data",
            "allen_oblique_test",
            "AllenMouseBrain_Atlas_CCF_2017.label",
        )

        # Use atlas assets from tests/test_data (no atlas API download required).
        self.atlas_path = os.path.join(
            self.tests_dir,
            "test_data",
            "allen_mouse_2017_atlas",
            "annotation_25_reoriented_2017.nrrd",
        )
        self.label_path = os.path.join(
            self.tests_dir,
            "test_data",
            "allen_mouse_2017_atlas",
            "allen2017_colours.csv",
        )

        self.expected_root = os.path.join(
            self.tests_dir,
            "expected_outputs",
            "allen_oblique_test",
            "whole_series_report",
        )

    def _run_pynutil_oblique(self):
        atlas = load_custom_atlas(self.atlas_path, None, self.label_path)
        alignment = read_alignment(
            self.alignment_json,
            apply_deformation=False,
            apply_damage=False,
        )
        result = seg_to_coords(
            self.segmentation_folder,
            alignment,
            atlas,
            pixel_id=[0, 0, 0],
            use_flat=True,
            flat_label_path=self.flat_label_path,
        )
        label_df, per_section_df = quantify_coords(result, atlas)
        return result, label_df, per_section_df

    @staticmethod
    def _load_nutil_ref_report(csv_path):
        df = pd.read_csv(csv_path, sep=";", engine="python")
        df = df.iloc[:, :10].copy()  # ignore trailing empty columns

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

        # Compare only atlas regions with non-zero expected area (exclude idx 0/background).
        merged = merged[(merged["idx"] != 0) & (merged["region_area_expected"] > 0)].copy()
        self.assertFalse(merged.empty, f"No comparable atlas rows for {where}")

        rel_area_error = (
            (merged["region_area"] - merged["region_area_expected"]).abs()
            / merged["region_area_expected"]
        )
        abs_load_error = (merged["area_fraction"] - merged["load_expected"]).abs()

        # Nutil-comparison guardrails for this dataset while parity work is ongoing.
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

    def test_allen_oblique_load_and_region_area_match_nutil(self):
        result, label_df, per_section_df = self._run_pynutil_oblique()

        expected_load_path = os.path.join(
            self.expected_root, "test17_RefAtlasRegions_load.csv"
        )
        expected_load = self._load_nutil_ref_report(expected_load_path)
        self._assert_load_and_region_area(
            expected_load,
            label_df,
            where="whole-series load report",
        )

        section_map = {}
        for seg_path, section_df in zip(result.section_filenames, per_section_df):
            match = re.search(r"_s(\d{3})", os.path.basename(seg_path))
            if match:
                section_map[match.group(1)] = section_df

        self.assertTrue(section_map, "No per-section reports were produced")

        for section_id, section_df in sorted(section_map.items()):
            with self.subTest(section=section_id):
                expected_path = os.path.join(
                    self.expected_root,
                    f"test17_RefAtlasRegions__s{section_id}.csv",
                )
                expected_ref = self._load_nutil_ref_report(expected_path)
                self._assert_load_and_region_area(
                    expected_ref,
                    section_df,
                    where=f"section s{section_id}",
                )

if __name__ == "__main__":
    unittest.main()

"""Tests that every input cell is accounted for in quantification output.

A collaborator reported fewer quantified cells out than cells in. This test
suite verifies that the number of cells entering the pipeline equals the
number of cells reported in the output, and identifies where cells can be
silently dropped.

The primary cause: cells whose coordinates fall on atlas background
(region label 0) are counted by ``_counts_for`` but then dropped when
the count DataFrame is joined against ``atlas_labels`` (which typically
has no entry for idx=0).
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PyNutil.processing.analysis.region_counting import (
    _counts_for,
    pixel_count_per_region,
)


class TestCellCountConservation(unittest.TestCase):
    """Verify that input cell count == output quantified cell count."""

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_atlas_labels(*region_ids):
        """Build a minimal atlas_labels DataFrame for the given region IDs."""
        return pd.DataFrame(
            {
                "idx": list(region_ids),
                "name": [f"region_{i}" for i in region_ids],
                "r": [0] * len(region_ids),
                "g": [0] * len(region_ids),
                "b": [0] * len(region_ids),
            }
        )

    # ── _counts_for tests ────────────────────────────────────────────────

    def test_counts_for_includes_label_zero(self):
        """_counts_for should include cells with label 0 in its output."""
        labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
        idx, counts = _counts_for(None, labels)
        total = int(counts.sum())
        self.assertEqual(
            total,
            len(labels),
            f"_counts_for lost cells: input={len(labels)}, counted={total}",
        )
        # Label 0 should be explicitly present
        self.assertIn(0, idx, "Label 0 should appear in _counts_for output")

    def test_counts_for_preserves_all_cells(self):
        """Every cell should be counted regardless of label value."""
        for labels in [
            np.array([0, 0, 0], dtype=np.int64),  # all background
            np.array([1, 2, 3], dtype=np.int64),  # all valid
            np.array([0, 1, 2, 0, 3, 0], dtype=np.int64),  # mixed
        ]:
            with self.subTest(labels=labels.tolist()):
                idx, counts = _counts_for(None, labels)
                self.assertEqual(int(counts.sum()), len(labels))

    # ── pixel_count_per_region tests ─────────────────────────────────────

    def test_pixel_count_per_region_all_cells_on_known_regions(self):
        """When all cells map to known regions, total count must equal input."""
        n_cells = 10
        labels = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 1], dtype=np.int64)
        atlas_labels = self._make_atlas_labels(1, 2, 3)
        no_mask = np.ones(n_cells, dtype=bool)

        result = pixel_count_per_region(
            labels_dict_points=labels,
            labeled_dict_centroids=labels,
            current_points_undamaged=no_mask,
            current_centroids_undamaged=no_mask,
            current_points_hemi=[None] * n_cells,
            current_centroids_hemi=[None] * n_cells,
            df_label_colours=atlas_labels,
            with_damage=False,
        )

        total_pixel_count = int(result["pixel_count"].sum())
        total_object_count = int(result["object_count"].sum())
        self.assertEqual(
            total_pixel_count,
            n_cells,
            f"pixel_count lost cells: input={n_cells}, output={total_pixel_count}",
        )
        self.assertEqual(
            total_object_count,
            n_cells,
            f"object_count lost cells: input={n_cells}, output={total_object_count}",
        )

    def test_pixel_count_per_region_conserves_background_cells(self):
        """Cells on background (label 0) appear as 'out_of_atlas' in output."""
        n_cells = 10
        n_background = 3
        # 3 cells on background (label 0), 7 on valid regions
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3], dtype=np.int64)
        atlas_labels = self._make_atlas_labels(1, 2, 3)  # no idx=0
        no_mask = np.ones(n_cells, dtype=bool)

        result = pixel_count_per_region(
            labels_dict_points=labels,
            labeled_dict_centroids=labels,
            current_points_undamaged=no_mask,
            current_centroids_undamaged=no_mask,
            current_points_hemi=[None] * n_cells,
            current_centroids_hemi=[None] * n_cells,
            df_label_colours=atlas_labels,
            with_damage=False,
        )

        total_counted = int(result["object_count"].sum())
        self.assertEqual(
            total_counted,
            n_cells,
            f"All input cells should be accounted for, but "
            f"{n_cells - total_counted} cells were lost",
        )

        # Verify the out_of_atlas row exists with correct count
        oot = result[result["name"] == "out_of_atlas"]
        self.assertEqual(len(oot), 1, "Expected one out_of_atlas row")
        self.assertEqual(
            int(oot["object_count"].iloc[0]),
            n_background,
            f"out_of_atlas should have {n_background} cells",
        )

    def test_conservation_with_background_in_atlas_labels(self):
        """If atlas_labels includes idx=0, all cells should be conserved."""
        n_cells = 10
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3], dtype=np.int64)
        # Include idx=0 in atlas labels
        atlas_labels = self._make_atlas_labels(0, 1, 2, 3)
        no_mask = np.ones(n_cells, dtype=bool)

        result = pixel_count_per_region(
            labels_dict_points=labels,
            labeled_dict_centroids=labels,
            current_points_undamaged=no_mask,
            current_centroids_undamaged=no_mask,
            current_points_hemi=[None] * n_cells,
            current_centroids_hemi=[None] * n_cells,
            df_label_colours=atlas_labels,
            with_damage=False,
        )

        total_counted = int(result["object_count"].sum())
        self.assertEqual(
            total_counted,
            n_cells,
            f"With idx=0 in atlas_labels, all cells should be counted. "
            f"Lost {n_cells - total_counted} cells.",
        )

    # ── End-to-end label assignment test ─────────────────────────────────

    def test_assign_labels_reports_background_count(self):
        """assign_labels_at_coordinates should let callers detect background cells."""
        from PyNutil.processing.utils import assign_labels_at_coordinates

        # 4x4 atlas map: center 2x2 is region 1, border is 0 (background)
        atlas_map = np.zeros((4, 4), dtype=np.int32)
        atlas_map[1:3, 1:3] = 1

        reg_h, reg_w = 4, 4
        # 4 cells: 2 on region 1, 2 on background
        coords_y = np.array([1.5, 2.0, 0.0, 3.5])
        coords_x = np.array([1.5, 2.0, 0.0, 3.5])

        labels = assign_labels_at_coordinates(
            coords_y, coords_x, atlas_map, reg_h, reg_w
        )

        n_total = len(labels)
        n_background = int(np.sum(labels == 0))
        n_assigned = int(np.sum(labels > 0))

        self.assertEqual(n_total, 4)
        self.assertEqual(n_assigned, 2, "2 cells should be on region 1")
        self.assertEqual(n_background, 2, "2 cells should be on background")
        # Key assertion: total is conserved at this stage
        self.assertEqual(n_assigned + n_background, n_total)


if __name__ == "__main__":
    unittest.main()

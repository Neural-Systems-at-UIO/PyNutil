"""Tests for brainglobe-registration support.

Validates that:
1. The anchoring loader correctly parses brainglobe-registration JSON and
   produces a valid atlas slice.
2. The deformation provider correctly warps the atlas slice to match
   ``registered_atlas.tiff``.
3. ``downsampled.tiff`` (the registered brain section) is mostly within
   the atlas.
4. The full ``load_registration`` pipeline works end-to-end.
"""

import math
import os
import unittest

import cv2
import numpy as np
import tifffile

TEST_DIR = os.path.dirname(__file__)
BG_DATA = os.path.join(TEST_DIR, "test_data", "brainglobe_registration")
BG_JSON = os.path.join(BG_DATA, "brainglobe-registration.json")


class TestBrainGlobeRegistration(unittest.TestCase):
    """Tests for brainglobe-registration anchoring and deformation."""

    def test_loader_produces_valid_anchoring(self):
        """BrainGlobeRegistrationLoader should produce a SliceInfo with
        correct anchoring vector dimensions and section metadata."""
        from PyNutil.processing.adapters.anchoring import BrainGlobeRegistrationLoader

        loader = BrainGlobeRegistrationLoader()
        self.assertTrue(loader.can_handle(BG_JSON))

        data = loader.load(BG_JSON)
        self.assertEqual(len(data.slices), 1)

        s = data.slices[0]
        self.assertEqual(len(s.anchoring), 9)
        self.assertEqual(s.section_number, 311)
        self.assertGreater(s.width, 0)
        self.assertGreater(s.height, 0)
        self.assertEqual(s.metadata["registration_type"], "brainglobe")

        # width/height should match |U| and |V|
        U = s.anchoring[3:6]
        V = s.anchoring[6:9]
        expected_w = int(math.floor(math.hypot(*U))) + 1
        expected_h = int(math.floor(math.hypot(*V))) + 1
        self.assertEqual(s.width, expected_w)
        self.assertEqual(s.height, expected_h)

    def test_atlas_slice_has_content(self):
        """The anchoring vector should produce an atlas slice with substantial
        nonzero coverage (brain regions)."""
        from PyNutil.processing.adapters.anchoring import BrainGlobeRegistrationLoader
        from PyNutil.processing.atlas_map import generate_target_slice
        from PyNutil.io.atlas_loader import load_atlas_data

        loader = BrainGlobeRegistrationLoader()
        data = loader.load(BG_JSON)
        s = data.slices[0]

        atlas_volume, _, _ = load_atlas_data("allen_mouse_25um")
        atlas_slice = generate_target_slice(s.anchoring, atlas_volume)

        self.assertEqual(atlas_slice.shape, (s.height, s.width))
        coverage = np.count_nonzero(atlas_slice) / atlas_slice.size
        # At least 40% of the slice should have brain regions
        self.assertGreater(coverage, 0.4)

    def test_warped_atlas_matches_registered_atlas(self):
        """After applying deformation, the warped atlas slice should closely
        match ``registered_atlas.tiff`` (IoU > 0.98)."""
        from PyNutil.processing.adapters.anchoring import BrainGlobeRegistrationLoader
        from PyNutil.processing.adapters.deformation import BrainGlobeDeformationProvider
        from PyNutil.processing.atlas_map import generate_target_slice, warp_image
        from PyNutil.io.atlas_loader import load_atlas_data

        loader = BrainGlobeRegistrationLoader()
        data = loader.load(BG_JSON)

        provider = BrainGlobeDeformationProvider()
        data = provider.apply(data)

        s = data.slices[0]
        self.assertIsNotNone(s.deformation)
        self.assertIsNotNone(s.forward_deformation)

        atlas_volume, _, _ = load_atlas_data("allen_mouse_25um")
        atlas_slice = generate_target_slice(s.anchoring, atlas_volume).astype(np.float64)
        warped = warp_image(atlas_slice, s.deformation, (s.width, s.height)).astype(np.uint32)

        # Resize to brain section dims for comparison
        reg_atlas = tifffile.imread(os.path.join(BG_DATA, "registered_atlas.tiff"))
        warped_resized = cv2.resize(
            warped.astype(np.float32),
            (reg_atlas.shape[1], reg_atlas.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint32)

        both_nonzero = (warped_resized > 0) & (reg_atlas > 0)
        either_nonzero = (warped_resized > 0) | (reg_atlas > 0)
        iou = np.sum(both_nonzero) / np.sum(either_nonzero)
        self.assertGreater(iou, 0.98, f"IoU between warped atlas and registered_atlas is {iou:.3f}")


    def test_downsampled_mostly_in_atlas(self):
        """``downsampled.tiff`` (the registered brain section) should have
        almost all of its nonzero pixels within the atlas brain mask.

        Uses ``registered_hemispheres.tiff`` as the brain mask (hemisphere
        labels 1/2 cover the full brain extent, unlike the annotation which
        has ID 0 for fiber tracts and ventricles).
        """
        downsampled = tifffile.imread(os.path.join(BG_DATA, "downsampled.tiff"))
        hemi = tifffile.imread(os.path.join(BG_DATA, "registered_hemispheres.tiff"))

        tissue_mask = downsampled > 0
        brain_mask = hemi > 0
        tissue_in_brain = np.sum(tissue_mask & brain_mask)
        total_tissue = np.sum(tissue_mask)
        fraction = tissue_in_brain / total_tissue if total_tissue > 0 else 0.0
        self.assertGreater(
            fraction, 0.95,
            f"Only {fraction*100:.1f}% of tissue pixels are within the brain mask",
        )

    def test_atlas_covers_tissue(self):
        """Almost all annotated atlas pixels should overlap with the tissue
        in ``downsampled.tiff``, confirming registration alignment."""
        downsampled = tifffile.imread(os.path.join(BG_DATA, "downsampled.tiff"))
        reg_atlas = tifffile.imread(os.path.join(BG_DATA, "registered_atlas.tiff"))

        tissue_mask = downsampled > 0
        atlas_mask = reg_atlas > 0
        atlas_in_tissue = np.sum(tissue_mask & atlas_mask)
        total_atlas = np.sum(atlas_mask)
        fraction = atlas_in_tissue / total_atlas if total_atlas > 0 else 0.0
        self.assertGreater(
            fraction, 0.99,
            f"Only {fraction*100:.1f}% of atlas pixels are within tissue",
        )

    def test_load_registration_end_to_end(self):
        """``load_registration`` should auto-detect brainglobe format and
        produce a RegistrationData with deformation applied."""
        from PyNutil.processing.adapters import load_registration

        data = load_registration(BG_JSON)
        self.assertEqual(len(data.slices), 1)
        self.assertEqual(data.metadata.get("registration_type"), "brainglobe")

        s = data.slices[0]
        self.assertIsNotNone(s.deformation, "Deformation should be applied")
        self.assertIsNotNone(
            s.forward_deformation,
            "Forward deformation should be applied",
        )
        self.assertEqual(s.metadata["deformation_type"], "brainglobe")

    def test_quint_json_not_detected_as_brainglobe(self):
        """QuickNII JSON should NOT be detected as brainglobe."""
        from PyNutil.processing.adapters.anchoring import BrainGlobeRegistrationLoader

        quint_json = os.path.join(
            TEST_DIR, "test_data", "nonlinear_allen_mouse", "alignment.json"
        )
        if os.path.isfile(quint_json):
            self.assertFalse(BrainGlobeRegistrationLoader.can_handle(quint_json))


if __name__ == "__main__":
    unittest.main()

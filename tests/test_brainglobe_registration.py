import importlib.util
import os
import pathlib
import sys
import types
import unittest

import numpy as np
import tifffile


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_registry_module():
    """Load adapters.registry without importing top-level PyNutil package."""
    pkg = types.ModuleType("PyNutil")
    pkg.__path__ = [str(REPO_ROOT / "PyNutil")]
    sys.modules.setdefault("PyNutil", pkg)

    processing_pkg = types.ModuleType("PyNutil.processing")
    processing_pkg.__path__ = [str(REPO_ROOT / "PyNutil" / "processing")]
    sys.modules.setdefault("PyNutil.processing", processing_pkg)

    adapters_pkg = types.ModuleType("PyNutil.processing.adapters")
    adapters_pkg.__path__ = [str(REPO_ROOT / "PyNutil" / "processing" / "adapters")]
    sys.modules.setdefault("PyNutil.processing.adapters", adapters_pkg)

    def _load(module_name: str, relative_path: str):
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(
            module_name, REPO_ROOT / relative_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    _load("PyNutil.processing.adapters.base", "PyNutil/processing/adapters/base.py")
    _load(
        "PyNutil.processing.adapters.visualign_deformations",
        "PyNutil/processing/adapters/visualign_deformations.py",
    )
    _load(
        "PyNutil.processing.adapters.deformation",
        "PyNutil/processing/adapters/deformation.py",
    )
    _load("PyNutil.processing.adapters.damage", "PyNutil/processing/adapters/damage.py")
    _load(
        "PyNutil.processing.adapters.anchoring",
        "PyNutil/processing/adapters/anchoring.py",
    )
    return _load(
        "PyNutil.processing.adapters.registry", "PyNutil/processing/adapters/registry.py"
    )


class TestBrainGlobeRegistration(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.join(
            os.path.dirname(__file__), "test_data", "brainglobe_registration_output"
        )
        self.registration_json = os.path.join(
            self.base_dir, "brainglobe-registration.json"
        )
        self.registry = _load_registry_module()

    def test_loads_brainglobe_registration_as_single_slice(self):
        data = self.registry.load_registration(
            self.registration_json, apply_deformation=False
        )

        self.assertEqual(len(data.slices), 1)
        section = data.slices[0]

        self.assertEqual(section.width, 428)
        self.assertEqual(section.height, 318)

        expected_anchoring = np.array(
            [320.0, -33.0, -9.0, -17.0, -2.0, 475.0, -63.0, 394.0, 0.0],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            np.asarray(section.anchoring, dtype=np.float64),
            expected_anchoring,
            rtol=0,
            atol=1e-6,
        )

    def test_registered_intensity_is_mostly_within_registered_atlas_mask(self):
        image = tifffile.imread(os.path.join(self.base_dir, "downsampled.tiff"))
        hemispheres = tifffile.imread(
            os.path.join(self.base_dir, "registered_hemispheres.tiff")
        )

        foreground = image > 10
        inside_mask = hemispheres > 0

        coverage = float(np.mean(inside_mask[foreground]))
        self.assertGreaterEqual(coverage, 0.98)

    def test_nonlinear_deformation_fields_are_present_and_match_registration_shape(self):
        field0 = tifffile.imread(os.path.join(self.base_dir, "deformation_field_0.tiff"))
        field1 = tifffile.imread(os.path.join(self.base_dir, "deformation_field_1.tiff"))

        self.assertEqual(field0.shape, (318, 428))
        self.assertEqual(field1.shape, (318, 428))
        self.assertTrue(np.isfinite(field0).all())
        self.assertTrue(np.isfinite(field1).all())


if __name__ == "__main__":
    unittest.main()

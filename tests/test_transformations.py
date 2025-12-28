import unittest

import numpy as np

from PyNutil.processing.transformations import transform_to_atlas_space


class TestTransformations(unittest.TestCase):
    def test_transform_to_atlas_space_vectorized_matches_scalar(self):
        # Regression test for a Python/numpy-dependent vectorization issue that
        # could produce wildly incorrect coordinates for large input arrays.
        reg_h, reg_w = 757, 1108
        anchoring = [
            9.07763055202696,
            170.70087741438195,
            342.9600061041366,
            452.4794936686928,
            17.480539026392947,
            -14.099620905569232,
            -9.933576244949542,
            6.343873331864044,
            -310.91936823254576,
        ]

        mask = np.ones((reg_h, reg_w), dtype=bool)
        sig_y, sig_x = np.where(mask)

        n = 100_000
        idx = np.linspace(0, sig_y.size - 1, n).astype(np.int64)
        y = sig_y[idx]
        x = sig_x[idx]

        pts = transform_to_atlas_space(anchoring, y, x, reg_h, reg_w)

        o = np.asarray(anchoring[0:3], dtype=np.float64)
        u = np.asarray(anchoring[3:6], dtype=np.float64)
        v = np.asarray(anchoring[6:9], dtype=np.float64)
        y_scale = y.astype(np.float64) / float(reg_h)
        x_scale = x.astype(np.float64) / float(reg_w)

        expected = (
            o[None, :]
            + x_scale[:, None] * u[None, :]
            + y_scale[:, None] * v[None, :]
        )

        # NOTE: numpy.testing.assert_allclose has been observed to mis-handle
        # shapes under some Python/numpy builds (broadcasting errors with empty
        # desired arrays). Use a direct check instead.
        ok = np.allclose(pts, expected, rtol=1e-12, atol=1e-12)
        if not ok:
            max_err = float(np.max(np.abs(pts - expected)))
            self.fail(f"transform_to_atlas_space mismatch (max abs err={max_err})")


if __name__ == "__main__":
    unittest.main()

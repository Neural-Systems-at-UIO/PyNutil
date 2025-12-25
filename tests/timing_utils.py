"""Utilities for printing per-test timings.

These are intentionally non-failing helpers used during optimization.
"""

from __future__ import annotations

import time
import unittest


class TimedTestCase(unittest.TestCase):
    """A unittest.TestCase that prints how long each test takes.

    This only reports timing; it does not affect pass/fail outcomes.
    """

    def run(self, result=None):  # noqa: ANN001 (unittest signature)
        start = time.perf_counter()
        try:
            return super().run(result)
        finally:
            duration_s = time.perf_counter() - start
            # Always print so `-q` still shows timings during optimization.
            # `self.id()` is stable and includes module/class/test name.
            print(f"[timing] {self.id()} {duration_s:.3f}s")

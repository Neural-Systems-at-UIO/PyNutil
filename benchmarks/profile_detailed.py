"""Detailed per-function profiling of PyNutil pipeline."""
import cProfile
import pstats
import io
import json
import math
import os
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark import (
    _write_scenario, ATLAS_SHAPE, COLOUR,
)
from PyNutil import PyNutil

def run(paths):
    pn = PyNutil(
        segmentation_folder=paths["seg_folder"],
        alignment_json=paths["alignment_json"],
        colour=COLOUR,
        atlas_path=paths["atlas_path"],
        label_path=paths["label_path"],
        hemi_path=paths["hemi_path"],
    )
    pn.get_coordinates(non_linear=True)
    pn.quantify_coordinates()

# Profile 10 images at 1000x1000
tmpdir = tempfile.mkdtemp(prefix="pynutil_prof_")
paths = _write_scenario(tmpdir, 10, 1000)

# Warmup
run(paths)

pr = cProfile.Profile()
pr.enable()
run(paths)
pr.disable()

for sort_key in ("tottime", "cumtime"):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
    ps.print_stats(50)
    print(f"\n{'='*80}\nSorted by {sort_key}\n{'='*80}")
    print(s.getvalue())

shutil.rmtree(tmpdir, ignore_errors=True)

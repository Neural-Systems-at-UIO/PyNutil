"""End-to-end benchmark for PyNutil.

Generates a synthetic atlas (NRRD), hemisphere map, label CSV, segmentation
images, and alignment JSON, then runs the full PyNutil pipeline
(init -> get_coordinates -> quantify_coordinates) and reports wall-clock time
and peak memory usage.

Each scenario runs in a **subprocess** so that peak-RSS is measured per
scenario rather than cumulatively across the whole process.

Scenarios: 1 / 5 / 10 images at 500x500 and 1000x1000 resolution.

Usage:
    python benchmarks/benchmark.py              # Markdown table to stdout
    python benchmarks/benchmark.py --json       # JSON output
    python benchmarks/benchmark.py --run-scenario <tmpdir>   # (internal)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import nrrd
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────

ATLAS_SHAPE = (200, 230, 160)  # (x, y, z) - small synthetic atlas
N_REGIONS = 20
IMAGE_COUNTS = [1, 5, 10]
RESOLUTIONS = [500, 1000]
CELLPOSE_EXTRA_SCENARIOS = [(5, 1000)]
WARMUP_RUNS = 1
TIMED_RUNS = 5
COLOUR = [0, 0, 0]


# ── Synthetic data builders ───────────────────────────────────────────────


def _make_atlas_volume(shape):
    """Synthetic annotation volume with ~N_REGIONS block regions."""
    rng = np.random.RandomState(42)
    vol = np.zeros(shape, dtype=np.int32)
    sx, sy, sz = shape
    for rid in range(1, N_REGIONS + 1):
        x0 = rng.randint(0, sx // 2)
        y0 = rng.randint(0, sy // 2)
        z0 = rng.randint(0, sz // 2)
        vol[x0 : min(x0 + rng.randint(20, sx // 3), sx),
            y0 : min(y0 + rng.randint(20, sy // 3), sy),
            z0 : min(z0 + rng.randint(20, sz // 3), sz)] = rid
    return vol


def _make_hemi_map(shape):
    """Left (1) / right (2) split along the X midline."""
    hemi = np.ones(shape, dtype=np.int32)
    hemi[shape[0] // 2:, :, :] = 2
    return hemi


def _make_labels_csv(path, n_regions=N_REGIONS):
    """Write a minimal atlas label CSV."""
    rng = np.random.RandomState(99)
    rows = [{"idx": 0, "name": "background", "r": 0, "g": 0, "b": 0}]
    for i in range(1, n_regions + 1):
        rows.append({
            "idx": i, "name": f"region_{i}",
            "r": int(rng.randint(0, 256)),
            "g": int(rng.randint(0, 256)),
            "b": int(rng.randint(0, 256)),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_segmentation(height, width, colour, seed=0):
    """Random blobs on a white background."""
    rng = np.random.RandomState(seed)
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    n_blobs = max(1, int(height * width * 0.02 / 500))
    for _ in range(n_blobs):
        cy, cx = rng.randint(0, height), rng.randint(0, width)
        radius = rng.randint(3, max(4, min(height, width) // 20))
        cv2.circle(img, (cx, cy), radius, tuple(int(c) for c in colour), -1)
    return img


def _make_cellpose_segmentation(height, width, seed=0):
    """Synthetic Cellpose-like labeled image with integer object IDs."""
    rng = np.random.RandomState(seed)
    seg = np.zeros((height, width), dtype=np.uint16)
    n_objs = max(1, int(height * width * 0.02 / 500))
    for obj_id in range(1, n_objs + 1):
        cy, cx = rng.randint(0, height), rng.randint(0, width)
        radius = rng.randint(3, max(4, min(height, width) // 20))
        cv2.circle(seg, (cx, cy), radius, int(obj_id), -1)
    return seg


def _make_anchoring(section_idx, n_sections, atlas_shape):
    """Coronal-like anchoring vector spread along the Y axis."""
    sx, sy, sz = atlas_shape
    oy = (section_idx + 0.5) / n_sections * sy
    return [0.0, oy, float(sz), float(sx), 0.0, 0.0, 0.0, 0.0, -float(sz)]


def _make_markers(reg_width, reg_height, seed=0):
    """Six VisuAlign-style marker pairs for non-linear deformation."""
    rng = np.random.RandomState(seed)
    markers = []
    for _ in range(6):
        x = rng.uniform(0.1, 0.9) * reg_width
        y = rng.uniform(0.1, 0.9) * reg_height
        dx = rng.uniform(-0.05, 0.05) * reg_width
        dy = rng.uniform(-0.05, 0.05) * reg_height
        markers.append([x, y, x + dx, y + dy])
    return markers


def _write_scenario(tmpdir, n_images, resolution, segmentation_format="binary"):
    """Write all synthetic files for one benchmark scenario.

    Returns:
        dict with paths and scenario metadata.
    """
    atlas_path = os.path.join(tmpdir, "atlas.nrrd")
    hemi_path = os.path.join(tmpdir, "hemi.nrrd")
    label_path = os.path.join(tmpdir, "labels.csv")

    atlas_vol = _make_atlas_volume(ATLAS_SHAPE)
    hemi_vol = _make_hemi_map(ATLAS_SHAPE)
    nrrd.write(atlas_path, atlas_vol)
    nrrd.write(hemi_path, hemi_vol)
    _make_labels_csv(label_path)

    seg_folder = os.path.join(tmpdir, "segmentations")
    os.makedirs(seg_folder, exist_ok=True)
    for i in range(n_images):
        if segmentation_format == "cellpose":
            img = _make_cellpose_segmentation(resolution, resolution, seed=i)
        else:
            img = _make_segmentation(resolution, resolution, COLOUR, seed=i)
        cv2.imwrite(os.path.join(seg_folder, f"bench_s{i + 1:03d}.png"), img)

    anchoring_0 = _make_anchoring(0, max(n_images, 1), ATLAS_SHAPE)
    ux, uy, uz = anchoring_0[3:6]
    vx, vy, vz = anchoring_0[6:9]
    reg_width = int(math.floor(math.hypot(ux, uy, uz))) + 1
    reg_height = int(math.floor(math.hypot(vx, vy, vz))) + 1

    slices = []
    for i in range(n_images):
        slices.append({
            "filename": f"bench_s{i + 1:03d}.png",
            "nr": i + 1,
            "width": resolution,
            "height": resolution,
            "anchoring": _make_anchoring(i, n_images, ATLAS_SHAPE),
            "markers": _make_markers(reg_width, reg_height, seed=i + 100),
        })

    alignment = {
        "name": "benchmark",
        "target": "synthetic_atlas",
        "target-resolution": list(ATLAS_SHAPE),
        "slices": slices,
    }
    alignment_path = os.path.join(tmpdir, "alignment.json")
    with open(alignment_path, "w") as f:
        json.dump(alignment, f)

    # Write scenario metadata so the subprocess knows the parameters.
    meta = {
        "mode": segmentation_format,
        "n_images": n_images,
        "resolution": resolution,
        "atlas_path": atlas_path,
        "hemi_path": hemi_path,
        "label_path": label_path,
        "seg_folder": seg_folder,
        "alignment_json": alignment_path,
    }
    with open(os.path.join(tmpdir, "meta.json"), "w") as f:
        json.dump(meta, f)

    return meta


# ── Subprocess runner (for accurate per-scenario peak memory) ─────────────


def _run_scenario_subprocess(tmpdir):
    """Launch this script with --run-scenario in a fresh process.

    Returns the JSON result dict from the subprocess.
    """
    result = subprocess.run(
        [sys.executable, __file__, "--run-scenario", tmpdir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(f"  subprocess failed:\n{result.stderr}\n")
        return None
    return json.loads(result.stdout)


def _run_scenario_in_process(tmpdir):
    """Run a scenario inside the current process (used by --run-scenario)."""
    # Ensure project root is importable.
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    from PyNutil import (  # noqa: local import
        load_custom_atlas, read_alignment, seg_to_coords, quantify_coords,
    )
    from PyNutil.io.loaders import load_json_file  # noqa: local import

    meta = load_json_file(os.path.join(tmpdir, "meta.json"))

    def _run():
        # Suppress stdout from PyNutil (e.g. "Found N segmentations")
        # by temporarily redirecting it to stderr.
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            atlas = load_custom_atlas(
                meta["atlas_path"], meta["hemi_path"], meta["label_path"],
            )
            alignment = read_alignment(meta["alignment_json"])
            coords = seg_to_coords(
                meta["seg_folder"],
                alignment,
                atlas,
                pixel_id=COLOUR,
                segmentation_format=meta.get("mode", "binary"),
            )
            quantify_coords(coords, atlas)
        finally:
            sys.stdout = old_stdout

    # Warmup
    for _ in range(WARMUP_RUNS):
        _run()
        gc.collect()

    # Timed runs
    times = []
    for _ in range(TIMED_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        _run()
        times.append(time.perf_counter() - t0)

    # Peak RSS for THIS process (fresh subprocess = per-scenario)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        peak_mb = usage.ru_maxrss / (1024 * 1024)
    else:
        peak_mb = usage.ru_maxrss / 1024

    avg = float(np.mean(times))
    n_images = meta["n_images"]
    return {
        "mode": meta.get("mode", "binary"),
        "n_images": n_images,
        "resolution": f"{meta['resolution']}x{meta['resolution']}",
        "total_s": round(avg, 4),
        "per_section_s": round(avg / n_images, 4),
        "peak_mem_mb": round(peak_mb, 1),
    }


# ── Output formatting ─────────────────────────────────────────────────────


def _format_markdown(results):
    """Return the Markdown table as a string."""
    lines = []
    binary_results = [r for r in results if r.get("mode", "binary") == "binary"]
    cellpose_results = [r for r in results if r.get("mode") == "cellpose"]

    resolutions = sorted(
        set(r["resolution"] for r in binary_results),
        key=lambda s: int(s.split("x")[0]),
    )
    n_images_list = sorted(set(r["n_images"] for r in binary_results))
    lookup = {(r["n_images"], r["resolution"]): r for r in binary_results}

    lines.append("## Benchmark Results\n")

    cols = []
    for n in n_images_list:
        cols += [f"{n} img total", f"{n} img/section", f"{n} img mem"]
    lines.append("| Resolution | " + " | ".join(cols) + " |")
    lines.append(
        "|" + "|".join("-" * (max(len(c), 4) + 2) for c in ["Resolution"] + cols) + "|"
    )

    for res in resolutions:
        cells = []
        for n in n_images_list:
            r = lookup.get((n, res))
            if r:
                cells.append(f"{r['total_s']:.3f}s")
                cells.append(f"{r['per_section_s']:.3f}s")
                cells.append(f"{r['peak_mem_mb']:.0f}MB")
            else:
                cells += ["N/A", "N/A", "N/A"]
        lines.append(
            f"| {res:>10} | " + " | ".join(f"{c:>12}" for c in cells) + " |"
        )

    if cellpose_results:
        lines.append("\n## Additional Cellpose Benchmark\n")
        lines.append("| Resolution | Images | total | per section | peak mem |")
        lines.append("|------------|--------|-------|-------------|----------|")
        for r in sorted(
            cellpose_results,
            key=lambda x: (int(x["resolution"].split("x")[0]), x["n_images"]),
        ):
            lines.append(
                f"| {r['resolution']:>10} | {r['n_images']:>6} | "
                f"{r['total_s']:.3f}s | {r['per_section_s']:.3f}s | {r['peak_mem_mb']:.0f}MB |"
            )

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PyNutil end-to-end benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    parser.add_argument(
        "--run-scenario", metavar="TMPDIR",
        help="(internal) Run a single scenario in this process and print JSON result",
    )
    args = parser.parse_args()

    # Internal: run a single scenario and exit.
    if args.run_scenario:
        result = _run_scenario_in_process(args.run_scenario)
        print(json.dumps(result))
        return

    # Orchestrator: generate data, launch subprocesses, collect results.
    results = []
    for n_images in IMAGE_COUNTS:
        for resolution in RESOLUTIONS:
            sys.stderr.write(
                f"  benchmarking {n_images} images @ {resolution}x{resolution}...\n"
            )
            tmpdir = tempfile.mkdtemp(prefix="pynutil_bench_")
            try:
                _write_scenario(tmpdir, n_images, resolution, segmentation_format="binary")
                row = _run_scenario_subprocess(tmpdir)
                if row:
                    results.append(row)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    for n_images, resolution in CELLPOSE_EXTRA_SCENARIOS:
        sys.stderr.write(
            f"  benchmarking cellpose {n_images} images @ {resolution}x{resolution}...\n"
        )
        tmpdir = tempfile.mkdtemp(prefix="pynutil_bench_cellpose_")
        try:
            _write_scenario(tmpdir, n_images, resolution, segmentation_format="cellpose")
            row = _run_scenario_subprocess(tmpdir)
            if row:
                results.append(row)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        md = _format_markdown(results)
        print(md)

        # Save to benchmarks directory
        out_dir = Path(__file__).resolve().parent
        md_path = out_dir / "results.md"
        md_path.write_text(md)
        json_path = out_dir / "results.json"
        json_path.write_text(json.dumps(results, indent=2) + "\n")
        sys.stderr.write(f"  saved {md_path} and {json_path}\n")


if __name__ == "__main__":
    main()

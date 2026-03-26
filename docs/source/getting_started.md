# Getting started

PyNutil requires Python 3.8 or above.

## Installation

Install the Python package from PyPI:

```bash
pip install PyNutil
```

If you are working from the repository and want to run the bundled demos, install
the package in editable mode from the repository root:

```bash
pip install -e .
```

PyNutil can be used with:

- BrainGlobe atlases from the [BrainGlobe Atlas API](https://github.com/brainglobe/brainglobe-atlasapi)
- Custom atlas volumes in `.nrrd` format, for example the sample data in `tests/test_data/allen_mouse_2017_atlas`

For the desktop GUI, download the Windows or macOS executable from the
[GitHub releases page](https://github.com/Neural-Systems-at-UIO/PyNutil/releases).

## Basic workflow

PyNutil expects:

1. An atlas
2. A corresponding alignment JSON created with QuickNII, VisuAlign, DeepSlice, or compatible tooling
3. A segmentation file for each brain section, with the feature of interest encoded as a unique RGB color

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

# Load an atlas (BrainGlobe) and alignment
atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment("path/to/alignment.json")

# Extract coordinates from segmentations
coords = pnt.seg_to_coords(
    "path/to/segmentations/",
    alignment,
    atlas,
    pixel_id=[0, 0, 0],
    # For cellpose segmentations: segmentation_format="cellpose"
)

# Quantify by atlas region
label_df = pnt.quantify_coords(coords, atlas)

# Save results
pnt.save_analysis("path/to/output", coords, atlas, label_df=label_df)
```

For custom atlases that are not provided by BrainGlobe, use
`pnt.load_custom_atlas()` instead of `BrainGlobeAtlas(...)`.

The quantification table returned by `pnt.quantify_coords(...)` is a pandas
DataFrame. A typical subset of columns for a segmentation-based workflow looks
like this:

| idx | name | region_area | object_count | area_fraction |
| --- | --- | ---: | ---: | ---: |
| 0 | Clear Label | 15234 | 0 | 0.0000 |
| 8 | Basic cell groups and regions | 8421 | 17 | 0.0315 |
| 567 | Cerebrum | 12984 | 42 | 0.0648 |
| 688 | Cerebral cortex | 10327 | 31 | 0.0521 |

The exact columns depend on the workflow:

- segmentation workflows commonly include `region_area`, `object_count`,
  `object_pixels`, `object_area`, and `area_fraction`
- intensity workflows commonly include `sum_intensity` and `mean_intensity`
- hemisphere-aware atlases add left/right hemisphere columns
- damage-aware inputs add damaged/undamaged columns

## BrainGlobe and custom atlases

With a BrainGlobe atlas:

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment("path/to/alignment.json")
```

With a custom atlas:

```python
import PyNutil as pnt

atlas = pnt.load_custom_atlas(
    atlas_path="path/to/annotation.nrrd",
    hemi_path=None,
    label_path="path/to/labels.csv",
)
alignment = pnt.read_alignment("path/to/alignment.json")
```

## Alternative input modes

### Intensity images

If you want to measure image intensity rather than segmented objects, use
`image_to_coords()` instead of `seg_to_coords()`:

```python
coords = pnt.image_to_coords(
    "path/to/images/",
    alignment,
    atlas,
)
label_df = pnt.quantify_coords(coords, atlas)
```

### Pre-extracted coordinates

If you already have coordinates in image space, provide them as a CSV with the
columns `X`, `Y`, `image_width`, `image_height`, and `section number`, then use
`xy_to_coords()`:

```python
coords = pnt.xy_to_coords(
    "path/to/coordinates.csv",
    alignment,
    atlas,
)
label_df = pnt.quantify_coords(coords, atlas)
```

The input CSV should contain these columns:

- `X`
- `Y`
- `image_width`
- `image_height`
- `section number`

## Worked examples

The repository includes several scripts in `demos/` that show how to run
PyNutil with different input types and registration sources. These examples
assume PyNutil is importable as an installed package:

```bash
pip install -e .
```

### Standard segmentation workflow

`demos/basic_example.py` demonstrates the default pipeline with:

- a BrainGlobe atlas
- an alignment JSON
- binary segmentation images
- region quantification
- optional 3D volume interpolation
- saved analysis outputs

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(alignment_json)

coords = pnt.seg_to_coords(
    segmentation_folder,
    alignment,
    atlas,
    pixel_id=[0, 0, 0],
    object_cutoff=0,
    segmentation_format="binary",
)

label_df = pnt.quantify_coords(coords, atlas)
pnt.interpolate_volume(
    segmentation_folder=segmentation_folder,
    alignment_json=alignment_json,
    colour=[0, 0, 0],
    atlas=atlas,
)
pnt.save_analysis(output_folder, coords, atlas, label_df=label_df)
```

### Custom atlas example

`demos/basic_example_custom_atlas.py` shows how to load atlas data from local
files instead of the BrainGlobe Atlas API.

Use this pattern when you have:

- an annotation volume in `.nrrd`
- an optional hemisphere volume
- a CSV of region labels and colors

```python
import PyNutil as pnt

atlas = pnt.load_custom_atlas(
    atlas_path="path/to/annotation.nrrd",
    hemi_path=None,
    label_path="path/to/labels.csv",
)
alignment = pnt.read_alignment("path/to/alignment.json")
coords = pnt.seg_to_coords(
    "path/to/segmentations",
    alignment,
    atlas,
    pixel_id=[0, 0, 0],
)
label_df = pnt.quantify_coords(coords, atlas)
pnt.save_analysis("path/to/output", coords, atlas, label_df=label_df)
```

### Intensity example

`demos/basic_example_intensity.py` demonstrates quantifying image intensity
rather than segmented objects.

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(alignment_json)
coords = pnt.image_to_coords(image_folder, alignment, atlas)
label_df = pnt.quantify_coords(coords, atlas)
pnt.save_analysis(output_folder, coords, atlas, label_df=label_df)
```

### Coordinate CSV example

`demos/coordinate_example.py` uses a CSV of already extracted image-space
coordinates. This is useful when segmentation or detection was done elsewhere
and you want PyNutil to handle registration, atlas-space transformation, and
quantification.

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment("path/to/brainglobe-registration.json")
coords = pnt.xy_to_coords("path/to/coordinates.csv", alignment, atlas)
label_df = pnt.quantify_coords(coords, atlas)
pnt.save_analysis("path/to/output", coords, atlas, label_df=label_df)
```

### BrainGlobe registration examples

`demos/brainglobe_registration_usage.py` shows how to use alignment output from
[brainglobe-registration](https://github.com/brainglobe/brainglobe-registration).
PyNutil expects the registration JSON and associated deformation files to live
in the same folder structure written by BrainGlobe registration.

`demos/brainglobe_coordinate_example.py` demonstrates the same registration
source with coordinate CSV input rather than segmentation images.

### Transform JSON example

`demos/using_transform_jsons.py` is a minimal variant of the segmentation
workflow using transform JSON data from QuickNII, VisuAlign, or
DeepSlice-compatible pipelines.

### Which example to start with

- Start with `basic_example.py` for standard binary segmentation quantification.
- Use `basic_example_custom_atlas.py` when your atlas is not distributed through BrainGlobe.
- Use `basic_example_intensity.py` when your signal is image intensity rather than objects.
- Use `coordinate_example.py` or `brainglobe_coordinate_example.py` when you already have point coordinates.
- Use `brainglobe_registration_usage.py` when your registration came from brainglobe-registration.

## Output files

PyNutil generates a set of reports and export files in the output folder you
specify.

### Per-hemisphere quantification

If your atlas includes a hemisphere map, PyNutil generates per-hemisphere
quantifications in addition to total values. BrainGlobe atlases include this
map by default, and PyNutil also writes hemisphere-specific point cloud files
for MeshView.

### Damage quantification

If damaged regions are marked with [QCAlign](https://www.nitrc.org/projects/qcalign),
PyNutil excludes those regions from point clouds and reports damaged and
undamaged measurements separately.

### MeshView JSON

PyNutil writes MeshView-compatible JSON files that can be opened in:

- [MeshView for the Allen Mouse](https://meshview.apps.ebrains.eu/?atlas=ABA_Mouse_CCFv3_2017_25um)
- [MeshView for the Waxholm Rat](https://meshview.apps.ebrains.eu/)

### Interpolated NIfTI volumes

If you interpolate the volume, PyNutil writes an interpolated NIfTI file that
can be viewed in:

- [siibra explorer](https://atlases.ebrains.eu/viewer/#/)
- [ITK-SNAP](https://github.com/pyushkevich/itksnap)

## Interpreting the tabular results

The main region-level report may contain the following columns:

| Column | Definition |
| --- | --- |
| `idx` | Atlas ID of the region |
| `name` | Atlas region name |
| `r`, `g`, `b` | RGB color values for the region |
| `Region area` | Area representing the region in the section |
| `Object count` | Number of disconnected objects in the region |
| `Object pixels` | Number of segmented pixels in the region |
| `Object area` | Area represented by segmented pixels |
| `Area fraction` | `Object pixels / Region area` |
| `Left hemi` | Per-hemisphere value for the left hemisphere |
| `Right hemi` | Per-hemisphere value for the right hemisphere |
| `Damaged` | Value restricted to damaged areas |
| `Undamaged` | Value restricted to undamaged areas |

For intensity-based measurements, object counts are replaced by intensity
metrics such as:

| Column | Definition |
| --- | --- |
| `Sum intensity` | Sum of all image pixels in a region |
| `Mean intensity` | Mean image intensity in a region |

## Related resources

- QUINT workflow: <https://quint-workflow.readthedocs.io/en/latest/>
- Feature requests and bug reports: <https://github.com/Neural-Systems-at-UIO/PyNutil/issues>
- Contact: `harry.carey95@gmail.com`

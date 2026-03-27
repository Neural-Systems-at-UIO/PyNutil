# Getting started

PyNutil is a Python toolkit for transforming section-based data into atlas
space and quantifying it by brain region.

PyNutil requires Python 3.8 or above.

## Installation

Install the Python package from PyPI:

```bash
pip install PyNutil
```

If you are working from the repository and want to run the bundled demos,
install the package in editable mode from the repository root:

```bash
pip install -e .
```

PyNutil can be used with:

- BrainGlobe atlases from the [BrainGlobe Atlas API](https://github.com/brainglobe/brainglobe-atlasapi)
- Custom atlas volumes in `.nrrd` format, for example the sample data in
  `tests/test_data/allen_mouse_2017_atlas`

If you want the desktop GUI instead of the Python API, download the Windows or
macOS executable from the
[GitHub releases page](https://github.com/Neural-Systems-at-UIO/PyNutil/releases).

## Choose your workflow

Most PyNutil runs follow the same pattern:

1. Load an atlas
2. Load a registration JSON with `read_alignment()`
3. Convert your input data into atlas-space coordinates
4. Quantify by atlas region with `quantify_coords()`
5. Save reports and exports with `save_analysis()`

Choose the coordinate extraction step based on your input data:

| Your input | Function to use | Typical use case |
| --- | --- | --- |
| Segmentation images | `seg_to_coords()` | Count labeled objects or pixels by region |
| Source images | `image_to_coords()` | Measure image intensity by region |
| DataFrame of detections | `xy_to_coords()` | Quantify points produced by another tool |

Choose the atlas based on where it comes from:

| Atlas source | How to load it |
| --- | --- |
| BrainGlobe atlas | `BrainGlobeAtlas("allen_mouse_25um")` |
| Local atlas files | `pnt.load_custom_atlas(...)` |

## First successful run

If you are running from the repository, the quickest way to verify that your
environment works is to reproduce the standard segmentation workflow using the
bundled test data:

```python
import os

from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

repo_root = os.path.abspath(".")
segmentation_folder = os.path.join(
    repo_root, "tests/test_data/nonlinear_allen_mouse/segmentations"
)
alignment_json = os.path.join(
    repo_root, "tests/test_data/nonlinear_allen_mouse/alignment.json"
)
output_folder = os.path.join(repo_root, "test_result/getting_started_example")

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(alignment_json)

result = pnt.seg_to_coords(
    segmentation_folder,
    alignment,
    atlas,
    pixel_id=[0, 0, 0],
    segmentation_format="binary",
)

label_df = pnt.quantify_coords(result, atlas)
pnt.save_analysis(output_folder, result, atlas, label_df=label_df)
```

This example does four things:

1. Loads a BrainGlobe atlas
2. Reads a registration JSON
3. Converts segmentation images into atlas-space coordinates
4. Writes the quantified output to `test_result/getting_started_example`

## Inputs PyNutil expects

### Atlas

PyNutil supports both BrainGlobe atlases and local custom atlases.

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

For custom atlases:

- `atlas_path` should point to an annotation volume, typically a `.nrrd` file
- `hemi_path` is optional and can be `None` if you do not have a hemisphere map
- `label_path` should point to a CSV containing region labels and colors
- the bundled sample file
  `tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv` includes
  columns such as `idx`, `name`, `r`, `g`, and `b`

### Registration JSON

Use `read_alignment()` to load the section-to-atlas alignment:

```python
alignment = pnt.read_alignment("path/to/alignment.json")
```

PyNutil will try to detect the registration format automatically. The same
entry point supports registration data produced by QuickNII, VisuAlign, and
BrainGlobe registration workflows.

By default, `read_alignment()` also tries to:

- apply non-linear deformation when the registration source provides it
- apply damage masks when they are available

If you want to disable those steps explicitly:

```python
alignment = pnt.read_alignment(
    "path/to/alignment.json",
    apply_deformation=False,
    apply_damage=False,
)
```

## Core workflows

### Segmentation images

Use `seg_to_coords()` when each section has a segmentation image and the class
of interest is encoded as a specific RGB value.

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment("path/to/alignment.json")

result = pnt.seg_to_coords(
    "path/to/segmentations/",
    alignment,
    atlas,
    pixel_id=[0, 0, 0],
    segmentation_format="binary",
)

label_df = pnt.quantify_coords(result, atlas)
pnt.save_analysis("path/to/output", result, atlas, label_df=label_df)
```

Common parameters:

- `pixel_id`: RGB value to quantify, for example `[0, 0, 0]`
- `segmentation_format="binary"`: standard binary segmentation images
- `segmentation_format="cellpose"`: Cellpose-style segmentation input
- `object_cutoff`: minimum object size to keep

### Intensity images

Use `image_to_coords()` when you want to quantify image intensity rather than
segmented objects.

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment("path/to/alignment.json")

result = pnt.image_to_coords(
    "path/to/images/",
    alignment,
    atlas,
)

label_df = pnt.quantify_coords(result, atlas)
pnt.save_analysis("path/to/output", result, atlas, label_df=label_df)
```

This workflow is useful when your input images contain signal intensity that
should be aggregated by atlas region instead of counted as discrete objects.

### Pre-extracted coordinates

Use `xy_to_coords()` when a different tool has already detected points in image
space. Pass the detections as a `pandas.DataFrame` with columns `X`, `Y`,
`image_width`, `image_height`, and `section number`.

```python
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment("path/to/alignment.json")
df = pd.read_csv("path/to/coordinates.csv")

result = pnt.xy_to_coords(
    df,
    alignment,
    atlas,
)

label_df = pnt.quantify_coords(result, atlas)
pnt.save_analysis("path/to/output", result, atlas, label_df=label_df)
```

The input CSV should contain these columns:

- `X`
- `Y`
- `image_width`
- `image_height`
- `section number`

The `section number` values must match the section numbers used in the
registration JSON.

## Outputs

`quantify_coords()` returns a pandas DataFrame. The exact columns depend on the
workflow and the atlas, but segmentation-based runs commonly include columns
like:

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

After `save_analysis()`, the output folder typically contains:

- `whole_series_report/counts.csv` for segmentation or coordinate workflows
- `whole_series_report/intensity.csv` for intensity workflows
- `whole_series_meshview/pixels_meshview.json`
- `whole_series_meshview/objects_meshview.json` when object-level outputs exist

## Interpolated 3D volumes

PyNutil can also project section-based data into a 3D atlas-space volume with
`interpolate_volume()`.

This is useful when you want:

- a 3D heatmap of segmented objects or image intensity
- a volume that can be viewed in downstream NIfTI tools
- a per-voxel sampling-frequency volume alongside the main signal volume

The function returns three arrays:

- `interpolated_volume`: the main reconstructed value volume
- `frequency_volume`: how many section-derived samples contributed to each voxel
- `damage_volume`: a binary volume marking damaged regions

Example:

```python
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")

gv, fv, dv = pnt.interpolate_volume(
    segmentation_folder="path/to/segmentations/",
    alignment_json="path/to/alignment.json",
    colour=[0, 0, 0],
    atlas=atlas,
    value_mode="pixel_count",
    segmentation_format="binary",
    segmentation_mode=True,
)

pnt.save_volume_niftis(
    output_folder="path/to/output",
    interpolated_volume=gv,
    frequency_volume=fv,
    damage_volume=dv,
    atlas_volume=atlas.annotation,
    voxel_size_um=atlas.voxel_size_um,
)
```

Common `value_mode` options:

- `pixel_count`: number of segmented pixels projected into each voxel
- `object_count`: number of segmented objects projected into each voxel
- `mean`: mean sampled intensity, useful for intensity-image workflows

If you are interpolating from source images instead of segmentation masks, set
`segmentation_mode=False` and point `segmentation_folder` to the image folder.
You can also use `intensity_channel`, `min_intensity`, and `max_intensity` to
control how intensity values are sampled.

`save_volume_niftis()` writes the generated volumes into:

- `interpolated_volume/interpolated_volume.nii.gz`
- `interpolated_volume/frequency_volume.nii.gz`
- `interpolated_volume/damage_volume.nii.gz`

These NIfTI exports are scaled to 8-bit on write and can be opened in tools
such as ITK-SNAP or siibra explorer.

## Worked examples

The repository includes several runnable scripts in `demos/` that show the same
patterns with real paths and test data. These examples assume PyNutil is
installed in the current environment:

```bash
pip install -e .
```

Useful starting points:

- `demos/basic_example.py`: standard segmentation workflow with a BrainGlobe atlas
- `demos/basic_example_custom_atlas.py`: segmentation workflow with a custom atlas
- `demos/basic_example_intensity.py`: intensity quantification workflow
- `demos/coordinate_example.py`: coordinate CSV workflow
- `demos/brainglobe_coordinate_example.py`: coordinate CSV workflow using BrainGlobe registration output
- `demos/brainglobe_registration_usage.py`: BrainGlobe registration examples

## Troubleshooting

If a first run does not behave as expected, these are the most common checks:

- If no objects are found, make sure `pixel_id` matches the RGB value present in
  the segmentation images
- If section files are skipped, make sure section numbers in filenames or CSV
  rows match the registration JSON
- If quantification looks incomplete, check whether damage masking is filtering
  points and whether that is desirable for your workflow
- If you are using a custom atlas, verify that the annotation volume and label
  table refer to the same region IDs

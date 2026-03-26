# GUI

PyNutil also includes a desktop GUI for running the main workflow without
writing Python code.

## Installation

Download the Windows or macOS executable from the
[PyNutil releases page](https://github.com/Neural-Systems-at-UIO/PyNutil/releases).

## What the GUI does

The GUI wraps the same core pipeline as the Python API:

- load a BrainGlobe atlas or a custom atlas
- read a registration JSON
- process either segmentation images or intensity images
- quantify the result by atlas region
- optionally build interpolated 3D volumes
- save reports and MeshView / NIfTI outputs

## Required inputs

The GUI expects:

1. A registration JSON
2. Either a segmentation folder or an image folder
3. An output folder
4. An atlas selection

For segmentation workflows you also choose the object color to quantify.

## Atlas choices

You can run the GUI with either:

- a BrainGlobe atlas selected by name
- a custom atlas defined by annotation and label files

BrainGlobe atlases can also be installed from the GUI workflow.

## Segmentation and intensity modes

The GUI supports two main analysis modes:

### Segmentation mode

Use this when your input folder contains segmentation images and you want to
count segmented structures by atlas region.

Typical inputs:

- registration JSON
- segmentation folder
- object color
- atlas

Typical outputs:

- `whole_series_report/counts.csv`
- MeshView point clouds
- optional interpolated volumes

### Intensity mode

Use this when your input folder contains source images and you want to measure
signal intensity by atlas region instead of counting segmented objects.

Typical inputs:

- registration JSON
- image folder
- atlas

Typical outputs:

- `whole_series_report/intensity.csv`
- MeshView exports using intensity values
- optional interpolated volumes

## Optional volume interpolation

The GUI can generate 3D volumes from the section data. When interpolation is
enabled, you can choose the value mode:

- `pixel_count`
- `mean`
- `object_count`

These outputs can be written as NIfTI files for downstream viewing.

## Output folders

The GUI writes the same output structure as the Python API. Common outputs
include:

- `whole_series_report/`
- `whole_series_meshview/`
- `interpolated_volume/`

Depending on the workflow, these may include:

| Output | Description |
| --- | --- |
| `counts.csv` | Region-level segmentation quantification |
| `intensity.csv` | Region-level intensity quantification |
| `pixels_meshview.json` | Atlas-space point cloud export |
| `objects_meshview.json` | Object-centroid point cloud export |
| `interpolated_volume.nii.gz` | Interpolated value volume |
| `frequency_volume.nii.gz` | Per-voxel sampling frequency |
| `damage_volume.nii.gz` | Binary damage mask volume |

## When to use the GUI vs Python

Use the GUI when you want:

- a no-code workflow
- quick interactive runs
- easy atlas and file selection

Use the Python API when you want:

- scripted or reproducible batch processing
- tighter control of the pipeline in code
- integration with other analysis workflows

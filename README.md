# PyNutil
PyNutil is under development. 

PyNutil is a Python library for brain-wide quantification and spatial analysis of features in serial section images from mouse and rat brain. It aims to replicate the Quantifier feature of the Nutil software (RRID: SCR_017183). It builds on registration to a standardised reference atlas with the QuickNII (RRID:SCR_016854) and VisuAlign software (RRID:SCR_017978) and feature extraction by segmentation with an image analysis software such as ilastik (RRID:SCR_015246). 

For more information about the QUINT workflow:
https://quint-workflow.readthedocs.io/en/latest/ 

# Available Atlases

PyNutil can be run using a custom atlas in .nrrd format (e.g. tests/test_data/Allen_mouse_2017_atlas)  

Pynutil can also be used with the atlases available in the [BrainGlobe_Atlas API](https://github.com/brainglobe/brainglobe-atlasapi). 

# Installation
## Python package
```
pip install PyNutil
```
## GUI
download the executable for windows and macOS via the [github releases tab](https://github.com/Neural-Systems-at-UIO/PyNutil/releases)

![image](https://github.com/user-attachments/assets/c3b3ca11-f0ef-4e2c-96d7-498ecf96514d)

# Usage

PyNutil requires Python 3.8 or above.

As input, PyNutil requires:
1. An atlas
2. A corresponding alignment JSON created with the QuickNII or VisuAlign software.
3. A segmentation file for each brain section with the features to be quantified displayed with a unique RGB colour code (it currently accepts many image formats: png, jpg, jpeg, etc).

```python
from PyNutil import PyNutil

"""
Here we define a quantifier object
The segmentations should be images which come out of ilastik, segmenting objects-of-interest
The alignment json should be from DeepSlice, QuickNII, or VisuAlign, it defines the sections position in an atlas
The colour says which colour is the object you want to quantify in your segmentation. It is defined in RGB
Finally the atlas name is the relevant atlas from brainglobe_atlasapi or a custom atlas in nrrd format.

basic_example.py (brainglobe_atlasapi)
basic_example_custom_atlas.py (custom atlas)

"""
pnt = PyNutil(
    segmentation_folder='../tests/test_data/non_linear_allen_mouse/segmentations/',
    alignment_json='../tests/test_data/non_linear_allen_mouse/alignment.json',
    colour=[0, 0, 0],
    atlas_name='allen_mouse_25um'
)

pnt.get_coordinates(object_cutoff=0)

pnt.quantify_coordinates()

pnt.save_analysis("PyNutil/test_result/myResults")
```
PyNutil generates a series of reports in the folder which you specify.

## Per-Hemisphere Quantification
If you use an atlas which has a hemisphere map (All brainglobe atlases have this, it is a volume in the shape of the atlas with 1 in the lft hemisphere and 2 in the right) PyNutil will generate per-hemisphere quantifications in addition to total numbers. In addition, PyNutil will also genearte additional per-hemisphere point cloud files for viewing in meshview.
## Damage Quantification
[The QCAlign tool](https://www.nitrc.org/projects/qcalign) allows you to mark damaged areas on your section. This means that these damaged areas are excluded from your point clouds. In addition, PyNutil will seperately quantify damaged and undamaged areas. Note the undamaged, and damaged column names. 
# Meshview json files
PyNutil will produce meshview json files. This can be opened in [MeshView for the Allen Mouse](https://meshview.apps.ebrains.eu/?atlas=ABA_Mouse_CCFv3_2017_25um) or for [the Waxholm Rat](https://meshview.apps.ebrains.eu/)

https://github.com/user-attachments/assets/d3a43ca9-133e-40d1-a1b9-9a359deabf2d 

# Interpreting the Results
Each column name has the following definition
| Column        | Definition                                                                          |
|---------------|--------------------------------------------------------------------------------------|
| idx           | The atlas ID of the region.                                                         |
| name          | The name of atlas region.                                                           |
| r             | The amount of red in the RGB value for for the region colour.                       |
| g             | The amount of green in the RGB value for for the region colour.                     |
| b             | The amount of blue in the RGB value for for the region colour.                      |
| Region area   | Area representing the region on the segmentation in pixel values.                   |
| Object count  | Number of objects located in the region. An object is a disconnected group of pixels|
| Object pixels | Number of pixels representing objects in this region.                               |
| Object area   | Area representing objects in this region (object pixels x pixel scale).             |
| Area fraction | Ratio of Object pixels to Region pixels (Object pixels / Region pixels).            |
| Left hemi     | For each of the other columns, what is that value for the left hemisphere alone     |
| Right hemi    | For each of the other columns, what is that value for the right hemisphere alone    |
| Damaged       | For each of the other columns, what is that value for the areas marked damaged alone|
| Undamaged     | For each of the other columns, what is that value for the areas marked undamaged alone|

 # Feature Requests
We are open to feature requests 😊 Simply open an issue in the github describing the feature you would like to see. 

# Acknowledgements
PyNutil is developed at the Neural Systems Laboratory at the Institute of Basic Medical Sciences, University of Oslo, Norway with support from the EBRAINS infrastructure, and funding support from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Framework Partnership Agreement No. 650003 (HBP FPA).

# Contributors
Harry Carey, Sharon C Yates, Gergely Csucs, Arda Balkir, Ingvild Bjerke, Rembrandt Bakker, Nicolaas Groeneboom, Maja A Puchades, Jan G Bjaalie.

# Licence
GNU General Public License v3

# Related articles
Yates SC, Groeneboom NE, Coello C, et al. & Bjaalie JG (2019) QUINT: Workflow for Quantification and Spatial Analysis of Features in Histological Images From Rodent Brain. Front. Neuroinform. 13:75. https://doi.org/10.3389/fninf.2019.00075

Groeneboom NE, Yates SC, Puchades MA and Bjaalie JG. Nutil: A Pre- and Post-processing Toolbox for Histological Rodent Brain Section Images. Front. Neuroinform. 2020,14:37. https://doi.org/10.3389/fninf.2020.00037

Puchades MA, Csucs G, Lederberger D, Leergaard TB and Bjaalie JG. Spatial registration of serial microscopic brain images to three-dimensional reference atlases with the QuickNII tool. PLosONE, 2019, 14(5): e0216796. https://doi.org/10.1371/journal.pone.0216796

Carey H, Pegios M, Martin L, Saleeba C, Turner A, Everett N, Puchades M, Bjaalie J, McMullan S. DeepSlice: rapid fully automatic registration of mouse brain imaging to a volumetric atlas. BioRxiv. https://doi.org/10.1101/2022.04.28.489953

Berg S., Kutra D., Kroeger T., Straehle C.N., Kausler B.X., Haubold C., et al. (2019) ilastik:interactive machine learning for (bio) image analysis. Nat Methods. 16, 1226–1232. https://doi.org/10.1038/s41592-019-0582-9

# Contact us
Report issues here on Github or email: support@ebrains.eu

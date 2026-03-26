from brainrender import Scene
from brainrender.actors import Points

from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(
    "tests/test_data/brainglobe_coordinates/brainglobe-registration.json"
)

coords = pnt.xy_to_coords(
    "tests/test_data/brainglobe_coordinates/coordinates.csv",
    alignment,
    atlas)


coordinates = coords.objects.points
coordinates *= 25 #resolution of atlas
coordinates


# Create a brainrender scene using the zebrafish atlas
scene = Scene(atlas_name="allen_mouse_25um", title="mouse")
scene.add(Points(coordinates, name="CELLS", colors="steelblue"))
# Render!
scene.render()
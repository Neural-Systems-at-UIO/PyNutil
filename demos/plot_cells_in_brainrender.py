from brainrender import Scene
from brainrender.actors import Points

from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(
    "tests/test_data/brainglobe_coordinates/registration/brainglobe-registration.json"
)

coords = pnt.xy_to_coords(
    "tests/test_data/brainglobe_coordinates/coordinates.csv",
    alignment,
    atlas)

def to_hex(rgb):
    """Convert an RGB triplet (0-255) to a hex string."""
    return '#%02x%02x%02x' % tuple(rgb)

coordinates = coords.objects.points
coordinates *= 25 #resolution of atlas
colours = [to_hex(atlas.structures[i]['rgb_triplet']) if i != 0 else '#000000' for i in coords.objects.labels]
scene = Scene(atlas_name="allen_mouse_25um", title="mouse")
scene.add(Points(coordinates, name="CELLS", colors=colours))
# Render!
scene.render()
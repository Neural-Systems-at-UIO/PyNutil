from PyNutil import PyNutil

pnt = PyNutil(
    image_folder="demo_data/brainglobe_registration_intensity_images",
    alignment_json="tests/test_data/brainglobe_registration/brainglobe-registration.json",
    atlas_name="allen_mouse_25um",
    intensity_channel="grayscale",
)

pnt.get_coordinates()
pnt.quantify_coordinates()
# Skip kNN interpolation to keep runtime reasonable for this one-off run.
pnt.interpolate_volume(do_interpolation=False)
pnt.save_analysis("demo_data/outputs/brainglobe_registration_intensity", create_visualisations=False)

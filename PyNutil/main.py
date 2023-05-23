from .metadata import metadata_loader
from .read_and_write import readAtlasVolume


class PyNutil:
    def __init__(
        self,
        segmentation_folder,
        json_file,
        colour,
        atlas,
    ) -> None:
        self.config, self.metadata_path = metadata_loader.load_config()
        if atlas not in self.config["annotation_volumes"]:
            raise ValueError(
                f"Atlas {atlas} not found in config file, valid atlases are: \n{' , '.join(list(self.config['annotation_volumes'].keys()))}"
            )
        self.segmentation_folder = segmentation_folder
        self.json_file = json_file
        self.colour = colour
        self.atlas = atlas
        # load the metadata json as well as the path to stored data files

    def build_quantifier(self):
        # do all the expensive computations
        atlas_root_path = self.config["annotation_volume_directory"]
        current_atlas_path = self.config["annotation_volumes"][self.atlas]["volume"]
        print("loading atlas volume")
        self.atlas_volume = readAtlasVolume(atlas_root_path + current_atlas_path)

    def get_coordinates(self):
        if not hasattr(self, "atlas_volume"):
            raise ValueError("Please run build_quantifier before running get_coordinates")

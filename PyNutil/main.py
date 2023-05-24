from .metadata import metadata_loader
from .read_and_write import readAtlasVolume
from .coordinate_extraction import FolderToAtlasSpaceMultiThreaded
import json

class PyNutil:
    def __init__(
        self,
        segmentation_folder=None,
        alignment_json=None,
        colour=None,
        volume_path=None,
        settings_file=None,
    ) -> None:
        self.config, self.metadata_path = metadata_loader.load_config()
        if settings_file is not None:
            with open(settings_file, "r") as f:
                settings = json.load(f)
            try:
                segmentation_folder = settings["segmentation_folder"]
                alignment_json = settings["alignment_json"]
                colour = settings["colour"]
                volume_path = settings["volume_path"]
            except KeyError as exc:
                raise KeyError(
                    "settings file must contain segmentation_folder, alignment_json, colour, and volume_path"
                ) from exc
        # check if any values are None
        if None in [segmentation_folder, alignment_json, colour, volume_path]:
            raise ValueError(
                "segmentation_folder, alignment_json, colour, and volume_path must all be specified and not be None"
            )
        if volume_path not in self.config["annotation_volumes"]:
            raise ValueError(
                f"Atlas {volume_path} not found in config file, valid atlases are: \n{' , '.join(list(self.config['annotation_volumes'].keys()))}"
            )
        
        self.segmentation_folder = segmentation_folder
        self.alignment_json = alignment_json
        self.colour = colour
        self.atlas = volume_path
        # load the metadata json as well as the path to stored data files

    def build_quantifier(self):
        # do all the expensive computations
        atlas_root_path = self.config["annotation_volume_directory"]
        current_atlas_path = self.config["annotation_volumes"][self.atlas]["volume"]
        print("loading atlas volume")
        self.atlas_volume = readAtlasVolume(atlas_root_path + current_atlas_path)
        print("atlas volume loaded")

    def get_coordinates(self, nonLinear=True, method="all"):
        if not hasattr(self, "atlas_volume"):
            raise ValueError("Please run build_quantifier before running get_coordinates")
        if method not in ["per_pixel", "per_object", "all"]:
            raise ValueError(f"method {method} not recognised, valid methods are: per_pixel, per_object, or all")
        print("extracting coordinates")
        points = FolderToAtlasSpaceMultiThreaded(
            self.segmentation_folder,
            self.alignment_json,
            pixelID=self.colour,
            nonLinear=nonLinear
        )
        self.points = points
        



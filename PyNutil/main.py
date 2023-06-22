from .metadata import metadata_loader
from .read_and_write import read_atlas_volume, write_points_to_meshview
from .coordinate_extraction import folder_to_atlas_space
from .counting_and_load import label_points, pixel_count_per_region
import json
import pandas as pd
from datetime import datetime
import os


class PyNutil:
    """A utility class for working with brain atlases and segmentation data.

    Parameters
    ----------
    segmentation_folder : str
        The path to the folder containing the segmentation data.
    alignment_json : str
        The path to the alignment JSON file.
    colour : int
        The colour of the segmentation data to extract.
    volume_path : str
        The name of the atlas volume to use.
    settings_file : str, optional
        The path to a JSON file containing the above parameters.

    Raises
    ------
    ValueError
        If any of the required parameters are None.

    Attributes
    ----------
    segmentation_folder : str
        The path to the folder containing the segmentation data.
    alignment_json : str
        The path to the alignment JSON file.
    colour : int
        The colour of the segmentation data to extract.
    atlas : str
        The name of the atlas volume being used.
    atlas_volume : numpy.ndarray
        The 3D array representing the atlas volume.
    atlas_labels : pandas.DataFrame
        A DataFrame containing the labels for the atlas volume.
    pixel_points : numpy.ndarray
        An array of pixel coordinates extracted from the segmentation data.
    labeled_points : numpy.ndarray
        An array of labeled pixel coordinates.
    label_df : pandas.DataFrame
        A DataFrame containing the pixel counts per region.

    Methods
    -------
    load_atlas_data()
        Loads the atlas volume and labels from disk.
    get_coordinates(non_linear=True, method='all')
        Extracts pixel coordinates from the segmentation data.
    extract_coordinates(non_linear, method)
        Extracts pixel coordinates from the segmentation data but is only used internally.
    quantify_coordinates()
        Quantifies the pixel coordinates by region.
    label_points()
        Labels the pixel coordinates by region but is only used internally.
    count_pixels_per_region(labeled_points)
        Counts the number of pixels per region but is only used internally.
    save_analysis(output_folder)
        Saves the pixel coordinates and pixel counts to disk.
    write_points_to_meshview(output_folder)
        Writes the pixel coordinates and labels to a JSON file for visualization but is only used internally.

    """

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
        self.atlas_volume, self.atlas_labels = self.load_atlas_data()

    def load_atlas_data(self):
        """Loads the atlas volume and labels from disk.

        Returns
        -------
        tuple
            A tuple containing the atlas volume as a numpy.ndarray and the atlas labels as a pandas.DataFrame.

        """
        # load the metadata json as well as the path to stored data files
        # this could potentially be moved into init
        atlas_root_path = self.config["annotation_volume_directory"]
        current_atlas_path = self.config["annotation_volumes"][self.atlas]["volume"]
        print("loading atlas volume")
        start_time = datetime.now()
        atlas_volume = read_atlas_volume(f"{atlas_root_path}{current_atlas_path}")
        time_taken = datetime.now() - start_time
        print(f"atlas volume loaded in: {time_taken} ✅")
        atlas_label_path = self.config["annotation_volumes"][self.atlas]["labels"]
        print("loading atlas labels")
        atlas_labels = pd.read_csv(f"{atlas_root_path}{atlas_label_path}")
        print("atlas labels loaded ✅")
        return atlas_volume, atlas_labels
    

    def get_coordinates(self, non_linear=True, method="all", object_cutoff=0):
        """Extracts pixel coordinates from the segmentation data.

        Parameters
        ----------
        non_linear : bool, optional
            Whether to use non-linear registration. Default is True.
        method : str, optional
            The method to use for extracting coordinates. Valid options are 'per_pixel', 'per_object', or 'all'.
            Default is 'all'.
        object_cutoff : int, optional
            The minimum number of pixels per object to be included in the analysis. Default is 1.

        Raises
        ------
        ValueError
            If the specified method is not recognized.

        """
        if not hasattr(self, "atlas_volume"):
            raise ValueError(
                "Please run build_quantifier before running get_coordinates"
            )
        if method not in ["per_pixel", "per_object", "all"]:
            raise ValueError(
                f"method {method} not recognised, valid methods are: per_pixel, per_object, or all"
            )
        print("extracting coordinates")
        (
            pixel_points,
            centroids,
            region_areas_list,
            points_len,
            centroids_len,
            segmentation_filenames,
        ) = folder_to_atlas_space(
            self.segmentation_folder,
            self.alignment_json,
            self.atlas_labels,
            pixel_id=self.colour,
            non_linear=non_linear,
            method=method,
            object_cutoff=object_cutoff,
        )
        self.pixel_points = pixel_points
        self.centroids = centroids
        ##points len and centroids len tell us how many points were extracted from each section
        ##This will be used to split the data up later into per section files
        self.points_len = points_len
        self.centroids_len = centroids_len
        self.segmentation_filenames = segmentation_filenames
        self.region_areas_list = region_areas_list

    def quantify_coordinates(self):
        """Quantifies the pixel coordinates by region.

        Raises
        ------
        ValueError
            If the pixel coordinates have not been extracted.

        """
        if not hasattr(self, "pixel_points") and not hasattr(self, "centroids"):
            raise ValueError(
                "Please run get_coordinates before running quantify_coordinates"
            )
        print("quantifying coordinates")
        labeled_points_centroids = None
        labeled_points = None
        if hasattr(self, "centroids"):
            labeled_points_centroids = label_points(
                self.centroids, self.atlas_volume, scale_factor=1
            )
        if hasattr(self, "pixel_points"):
            labeled_points = label_points(
                self.pixel_points, self.atlas_volume, scale_factor=1
            )

        prev_pl = 0
        prev_cl = 0
        per_section_df = []
        current_centroids = None
        current_points = None
        for pl,cl,ra in zip(self.points_len, self.centroids_len, self.region_areas_list):
            if hasattr(self, "centroids"):
                current_centroids = labeled_points_centroids[prev_cl : prev_cl + cl]
            if hasattr(self, "pixel_points"):
                current_points = labeled_points[prev_pl : prev_pl + pl]
            current_df = pixel_count_per_region(
                current_points, current_centroids, self.atlas_labels
            )

# create the df for section report and all report
# pixel_count_per_region returns a df with idx, pixel count, name and RGB.
# ra is region area list from 
# merge current_df onto ra (region_areas_list) based on idx column
#(left means use only keys from left frame, preserve key order)
            
            """
            Merge region areas and object areas onto the atlas label file.
            Remove duplicate columns
            Calculate and add area_fraction to new column in the df.  
            """
            all_region_df = self.atlas_labels.merge(ra, on = 'idx', how='left')           
            current_df_new = all_region_df.merge(current_df, on= 'idx', how= 'left', suffixes= (None,"_y")).drop(columns=["a","VIS", "MSH", "name_y","r_y","g_y","b_y"])
            current_df_new["area_fraction"] = current_df_new["pixel_count"] / current_df_new["region_area"]
            
            # Several alternatives for the merge code above
            """
            new_rows = []
            for index, row in all_region_df.iterrows():
                mask = current_df["idx"] == row ["idx"]
                current_region_row = current_df[mask]
                region_area = current_region_row["pixel_count"].values
                object_count = current_region_row["object_count"].values

                row["pixel_count"] = region_area[0]
                row["object_count"] = object_count[0]

                new_rows.append[row]
            
            current_df_new = pd.DataFrame(new_rows)
            """
            
            """
            new_rows = []
            for index, row in current_df.iterrows():
                mask = self.atlas_labels["idx"] == row["idx"]
                current_region_row = self.atlas_labels[mask]
                current_region_name = current_region_row["name"].values
                current_region_red = current_region_row["r"].values
                current_region_green = current_region_row["g"].values
                current_region_blue = current_region_row["b"].values

                row["name"] = current_region_name[0]
                row["r"] = current_region_red[0]
                row["g"] = current_region_green[0]
                row["b"] = current_region_blue[0]
                
                new_rows.append(row)

            current_df_new = pd.DataFrame(new_rows)"""
            
            #per_section_df.append(current_df_new)
            per_section_df.append(current_df_new)
            prev_pl += pl
            prev_cl += cl
        
        
        ##combine all the slice reports, groupby idx, name, rgb and sum region and object pixels. Remove area_fraction column and recalculate.
        self.label_df =  pd.concat(per_section_df).groupby(['idx','name','r','g','b']).sum().reset_index().drop(columns=['area_fraction'])
        self.label_df["area_fraction"] = self.label_df["pixel_count"] / self.label_df["region_area"]

        """
        Potential source of error:
        If there are duplicates in the label file, regional results will be duplicated and summed leading to incorrect results
        """

        #reorder the df to match the order of idx column in self.atlas_labels        
        self.label_df = self.label_df.set_index('idx')
        self.label_df = self.label_df.reindex(index=self.atlas_labels['idx'])
        self.label_df = self.label_df.reset_index()
        
        self.labeled_points = labeled_points
        self.labeled_points_centroids = labeled_points_centroids
        self.per_section_df = per_section_df

        print("quantification complete ✅")

    def save_analysis(self, output_folder):
        """Saves the pixel coordinates and pixel counts to different files in the specified
        output folder.

        Parameters
        ----------
        output_folder : str
            The path to the output folder.

        Raises
        ------
        ValueError
            If the pixel coordinates have not been extracted.

        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        if not os.path.exists(f"{output_folder}/sections_combined_report"):
            os.makedirs(f"{output_folder}/sections_combined_report")

        if not hasattr(self, "label_df"):
            print("no quantification found so we will only save the coordinates")
            print(
                "if you want to save the quantification please run quantify_coordinates"
            )
        else:
            
            self.label_df.to_csv(
                f"{output_folder}/sections_combined_report/counts.csv", sep=";", na_rep="", index=False
            )
        if not os.path.exists(f"{output_folder}/per_section_meshview"):
            os.makedirs(f"{output_folder}/per_section_meshview")
        if not os.path.exists(f"{output_folder}/per_section_reports"):
            os.makedirs(f"{output_folder}/per_section_reports")
        if not os.path.exists(f"{output_folder}/sections_combined_meshview"):
            os.makedirs(f"{output_folder}/sections_combined_meshview")

        prev_pl = 0
        prev_cl = 0

        for pl, cl, fn, df in zip(
            self.points_len,
            self.centroids_len,
            self.segmentation_filenames,
            self.per_section_df,
        ):
            split_fn = fn.split(os.sep)[-1].split(".")[0]
            df.to_csv(
                f"{output_folder}/per_section_reports/{split_fn}.csv",
                sep=";",
                na_rep="",
                index=False,
            )
            if hasattr(self, "pixel_points"):
                write_points_to_meshview(
                    self.pixel_points[prev_pl : pl + prev_pl],
                    self.labeled_points[prev_pl : pl + prev_pl],
                    f"{output_folder}/per_section_meshview/{split_fn}_pixels.json",
                    self.atlas_labels,
                )
            if hasattr(self, "centroids"):
                write_points_to_meshview(
                    self.centroids[prev_cl : cl + prev_cl],
                    self.labeled_points_centroids[prev_cl : cl + prev_cl],
                    f"{output_folder}/per_section_meshview/{split_fn}_centroids.json",
                    self.atlas_labels,
                )
            prev_cl += cl
            prev_pl += pl

        if hasattr(self, "pixel_points"):
            write_points_to_meshview(
                self.pixel_points,
                self.labeled_points,
                f"{output_folder}/sections_combined_meshview/pixels_meshview.json",
                self.atlas_labels,
            )
        if hasattr(self, "centroids"):
            write_points_to_meshview(
                self.centroids,
                self.labeled_points_centroids,
                f"{output_folder}/sections_combined_meshview/objects_meshview.json",
                self.atlas_labels,
            )
        print("analysis saved ✅")

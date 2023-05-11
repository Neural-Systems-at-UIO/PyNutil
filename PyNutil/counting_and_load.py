import numpy as np
import pandas as pd

# related to counting and load
def labelPoints(points, label_volume, scale_factor=1):
    """this function takes a list of points and assigns them to a region based on the regionVolume.
    These regions will just be the values in the regionVolume at the points.
    it returns a dictionary with the region as the key and the points as the value"""
    #first convert the points to 3 columns
    points = np.reshape(points, (-1,3))
    #scale the points
    points = points * scale_factor
    #round the points to the nearest whole number
    points = np.round(points).astype(int)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    #get the label value for each point
    labels = label_volume[x,y,z]
    return labels


# related to counting_and_load
def PixelCountPerRegion(labelsDict, label_colours): 
    """Function for counting no. of pixels per region and writing to CSV based on 
    a dictionary with the region as the key and the points as the value, """
    counted_labels, label_counts = np.unique(labelsDict, return_counts=True)
    # which regions have pixels, and how many pixels are there per region
    counts_per_label = list(zip(counted_labels,label_counts))
    # create a list of unique regions and pixel counts per region

    df_counts_per_label = pd.DataFrame(counts_per_label, columns=["allenID","pixel count"])
    # create a pandas df with regions and pixel counts

    df_label_colours =pd.read_csv(label_colours, sep=",")
    # find colours corresponding to each region ID and add to the pandas dataframe

    #look up name, r, g, b in df_allen_colours in df_counts_per_label based on "allenID"
    new_rows = []
    for index, row in df_counts_per_label.iterrows():
        mask = df_label_colours["allenID"] == row["allenID"] 
        current_region_row = df_label_colours[mask]
        current_region_name = current_region_row["name"].values
        current_region_red = current_region_row["r"].values
        current_region_green = current_region_row["g"].values
        current_region_blue = current_region_row["b"].values

        row["name"]  = current_region_name[0]
        row["r"] = current_region_red[0]
        row["g"] = current_region_green[0]
        row["b"] = current_region_blue[0]
        
        new_rows.append(row)

    df_counts_per_label_name = pd.DataFrame(new_rows)
    return df_counts_per_label_name


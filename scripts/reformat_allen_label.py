import pandas as pd

"""reformat itksnap_label_description_2022.txt"""
def reformat_allen_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep=" ", header=None,  names=["id", "r", "g", "b", "1a", "1b", "1c", "name"])
    
    # this is to reformat the name to allenID
    df[["name", "allenID"]] = df["name"].str.split(' - ', expand=True)
    
    # this is to add on "root" as this was missing from the Allen file
    df = df.append({"allenID": 0, "name": "background", "r": 255, "g": 255, "b": 255, "1a": 1.0, "1b":1.0, "1c":1.0}, ignore_index=True)
    df.to_csv(outputpath, index=False)

reformat_allen_label("../junk/itksnap_label_description_2022.txt","../junk/allen2022_colours.csv")


"""reformat AllenMouseBrain_atlas_CCF_2017.label"""
def reformat_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep = "\t", header=None, skiprows=15 ,names=["allenID", "r", "g", "b", "1a", "1b", "1c", "name"] )
    df = df.append({"allenID": 0, "name": "Clear Label", "r": 0, "g": 0, "b": 0, "1a": 1.0, "1b":1.0, "1c":1.0}, ignore_index=True)
    df.to_csv(outputpath, index=False)

reformat_label("../annotation_volumes/AllenMouseBrain_Atlas_CCF_2017.label","../annotation_volumes/allen2017_colours.csv")

"""reformat AllenMouseBrain_atlas_CCF_2015.label"""
def reformat_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep = "\t", header=None, skiprows=15 ,names=["allenID", "r", "g", "b", "1a", "1b", "1c", "name"] )
    df = df.append({"allenID": 0, "name": "Clear Label", "r": 0, "g": 0, "b": 0, "1a": 1.0, "1b":1.0, "1c":1.0}, ignore_index=True)
    df.to_csv(outputpath, index=False)

reformat_label("../annotation_volumes/AllenMouseBrain_Atlas_CCF_2015.label","../annotation_volumes/allen2015_colours.csv")

"""reformat WHS_SD_rat_atlas_v4"""
def reformat_WHS_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep = "\s+", header=None, skiprows=15 ,names=["allenID", "r", "g", "b", "1a", "1b", "1c", "name"] )
    df = df.append({"allenID": 0, "name": "Clear Label", "r": 0, "g": 0, "b": 0, "1a": 1.0, "1b":1.0, "1c":1.0}, ignore_index=True)
    df.to_csv(outputpath, index=False)

reformat_WHS_label("../annotation_volumes/WHS_SD_rat_atlas_v4.label","../annotation_volumes/WHS_v4_colours.csv")


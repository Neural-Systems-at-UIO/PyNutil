import pandas as pd


"""reformat itksnap_label_description_2022.txt"""
def reformat_allen_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep=" ", header=None,  names=["row", "r", "g", "b", "a", "VIS", "MSH", "name"])
    
    # this is to reformat the name to allenID
    df[["name", "idx"]] = df["name"].str.split(' - ', expand=True)
     # this is to add on "root" as this was missing from the Allen file
    df = df.append({"idx": 0, "name": "background", "r": 255, "g": 255, "b": 255, "a": 1.0, "VIS":1.0, "MSH":1.0}, ignore_index=True)
    
    df
    reordered_df = df.loc[:,["idx", "r", "g", "b", "a", "VIS", "MSH", "name", "row"]]
    reordered_df.to_csv(outputpath, index=False)

reformat_allen_label("../annotation_volumes/itksnap_label_description_2022.txt","../annotation_volumes/allen2022_colours_updated.csv")


"""reformat AllenMouseBrain_atlas_CCF_2017.label"""
def reformat_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep = "\t", header=None, skiprows=15 ,names=["idx", "r", "g", "b", "a", "VIS", "MSH", "name"] )
    df_clear = pd.DataFrame({"idx": 0, "name": "Clear Label", "r": 0, "g": 0, "b": 0, "a": 1.0, "VIS":1.0, "MSH":1.0}, ignore_index=True)
    df = pd.concat([df_clear, df])
    df.to_csv(outputpath, index=False)

reformat_label("../annotation_volumes/AllenMouseBrain_Atlas_CCF_2017.label","../annotation_volumes/allen2017_colours.csv")


"""reformat AllenMouseBrain_atlas_CCF_2015.label"""
def reformat_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep = "\t", header=None, skiprows=15 ,names=["idx", "r", "g", "b", "a", "VIS", "MSH", "name"] )
    df = df.append({"idx": 0, "name": "Clear Label", "r": 0, "g": 0, "b": 0, "a": 1.0, "VIS":1.0, "MSH":1.0}, ignore_index=True)
    df.to_csv(outputpath, index=False)

reformat_label("../annotation_volumes/AllenMouseBrain_Atlas_CCF_2015.label","../annotation_volumes/allen2015_colours.csv")


"""reformat WHS_rat_atlas"""
def reformat_WHS_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep = "\s+", header=None, skiprows=15 ,names=["idx", "r", "g", "b", "a", "VIS", "MSH", "name"] )
    df = df.append({"idx": 0, "name": "Clear Label", "r": 0, "g": 0, "b": 0, "a": 1.0, "VIS":1.0, "MSH":1.0}, ignore_index=True)
    df.to_csv(outputpath, index=False)

reformat_WHS_label("../annotation_volumes/WHS_SD_rat_atlas_v4.label","../annotation_volumes/WHS_v4_colours.csv")
reformat_WHS_label("../annotation_volumes/WHS_Atlas_Rat_Brain_v3.label","../annotation_volumes/WHS_v3_colours.csv")
reformat_WHS_label("../annotation_volumes/WHS_Atlas_Rat_Brain_v2.label","../annotation_volumes/WHS_v2_colours.csv")
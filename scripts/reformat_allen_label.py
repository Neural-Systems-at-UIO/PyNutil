import pandas as pd

def reformat_allen_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep=" ", header=None,  names=["id", "r", "g", "b", "1a", "1b", "1c", "name"])
    ##for reading gergely format labels try the skiprows argument (might be mispelt sorry). check the 
    ##read csv documentation for more details. 
    
    # this is to reformat the name to allenID
    df[["name", "allenID"]] = df["name"].str.split(' - ', expand=True)
    
    # this is to add on "root" as this was missing from the Allen file
    df = df.append({"allenID": 0, "name": "background", "r": 255, "g": 255, "b": 255}, ignore_index=True)
    df.to_csv(outputpath, index=False)

reformat_allen_label("../junk/itksnap_label_description_2022.txt","../junk/allen2022_colours.csv")


"""Task: Modify the function to reformat Gergely's label files"""

def reformat_label(inputpath, outputpath):
    df = pd.read_csv(inputpath, sep = " ", header=None)
    df.to_csv(outputpath, index=False)

reformat_label("../annotation_volumes/AllenMouseBrain_Atlas_CCF_2017.label","../annotation_volumes/allen2017_colours.csv")



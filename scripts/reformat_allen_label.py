import pandas as pd
#open colour txt file
path = "itksnap_label_description.txt"


# use " " as separator
#set column names
df = pd.read_csv(path, sep=" ", header=None,  names=["id", "r", "g", "b", "1a", "1b", "1c", "name"])
df[["name", "allenID"]] = df["name"].str.split(' - ', expand=True)
df = df.append({"allenID": 0, "name": "background", "r": 255, "g": 255, "b": 255}, ignore_index=True)
df.to_csv("allen2022_colours.csv", index=False)


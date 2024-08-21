import brainglobe_atlasapi
import pandas as pd
import nrrd
import matplotlib.pyplot as plt
import numpy as np
brainglobe_atlasapi.list_atlases.show_atlases()
allen = brainglobe_atlasapi.BrainGlobeAtlas('allen_mouse_25um')

keys = allen.structures.keys()



##current structure format 
labels = pd.read_csv(r'/home/harryc/github/PyNutil/PyNutil/metadata/annotation_volumes/allen2017_colours.csv')

labels


orig_vol = nrrd.read(r"/home/harryc/github/PyNutil/PyNutil/metadata/annotation_volumes/annotation_25_reoriented_2017.nrrd")

plt.imshow(orig_vol[0][200]>0)


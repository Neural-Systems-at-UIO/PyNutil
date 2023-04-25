"""
This script was needed to reorient the Allen Volume to match the 
orientation of meshview coordinates.
"""

import nrrd
import numpy as np
data, header = nrrd.read('../annotation_volumes/annotation_10.nrrd')
# change the order of the axes
data = np.transpose(data, (2,0,1))
# flip two of the axes
data = data[:, ::-1, ::-1]
# write the new volume
nrrd.write('../annotation_volumes/annotation_10_reoriented.nrrd', data, header=header)

import nrrd
import requests
import numpy as np
from tqdm import tqdm


def download_to_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)


def float_downsample_alternating(volume, pattern):
    def make_indices(max_dim, pattern):
        idx, current = [], 0
        i = 0
        while current < max_dim:
            idx.append(current)
            current += pattern[i % len(pattern)]  # cycle over entire pattern
            i += 1
        return np.array(idx)

    idx_z = make_indices(volume.shape[0], pattern)
    idx_y = make_indices(volume.shape[1], pattern)
    idx_x = make_indices(volume.shape[2], pattern)

    return volume[np.ix_(idx_z, idx_y, idx_x)]


url_template = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_{}/annotation_10.nrrd"
filename_template = "annotation_10_{}.nrrd"
data_dir = "../demo_data/"
years = [2015, 2016, 2017, 2022]
for year in tqdm(years):
    fn = f"{data_dir}/{filename_template.format(year)}"
    download_to_file(url_template.format(year), fn)
    allen_10, header_10 = nrrd.read(fn)
    allen_10 = np.transpose(allen_10, (2, 0, 1))
    # flip two of the axes
    allen_10 = allen_10[:, ::-1, ::-1]
    allen_25 = float_downsample_alternating(allen_10, [3, 2])
    nrrd.write(f"{data_dir}/reoriented_annotation_{year}_10um.nrrd", allen_10)
    nrrd.write(f"{data_dir}/reoriented_annotation_{year}_25um.nrrd", allen_25)

import nrrd
import requests
import numpy as np


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


url_100um_allen = r"https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_100.nrrd"
url_50um_allen = r"https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_50.nrrd"
url_25um_allen = r"https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd"
url_10um_allen = r"https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd"
download_to_file(url_100um_allen, "annotation_100.nrrd")
download_to_file(url_50um_allen, "annotation_50.nrrd")
download_to_file(url_25um_allen, "annotation_25.nrrd")
download_to_file(url_10um_allen, "annotation_10.nrrd")
allen_100, header_100 = nrrd.read("annotation_100.nrrd")
allen_50, header_50 = nrrd.read("annotation_50.nrrd")
allen_25, header_25 = nrrd.read("annotation_25.nrrd")
allen_10, header_10 = nrrd.read("annotation_10.nrrd")

downsampled_50_to_100 = float_downsample_alternating(allen_50, [2])
(downsampled_50_to_100 != allen_100).sum()

downsampled_10_to_25 = float_downsample_alternating(allen_10, [3, 2])
(downsampled_10_to_25 != allen_25).sum()

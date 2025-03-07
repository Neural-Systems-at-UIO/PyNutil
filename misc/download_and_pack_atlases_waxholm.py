import nibabel as nib
import nrrd
import requests
import numpy as np
from tqdm import tqdm
import pandas as pd

def download_to_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)

def reformat_WHS_label(inputpath, outputpath):
    df = pd.read_csv(
        inputpath,
        sep="\s+",
        header=None,
        skiprows=15,
        names=["idx", "r", "g", "b", "a", "VIS", "MSH", "name"],
    )
    # df = df.append({"idx": 0, "name": "Clear Label", "r": 0, "g": 0, "b": 0, "a": 1.0, "VIS":1.0, "MSH":1.0}, ignore_index=True)
    df_clear = pd.DataFrame(
        {
            "idx": 0,
            "name": "Clear Label",
            "r": 0,
            "g": 0,
            "b": 0,
            "a": 1.0,
            "VIS": 1.0,
            "MSH": 1.0,
        },
        index=[0],
    )
    df = pd.concat([df_clear, df])
    df.to_csv(outputpath, index=False)

label_urls = {
    "v4.01":"https://www.nitrc.org/frs/download.php/13399/WHS_SD_rat_atlas_v4.01.label//?i_agree=1&download_now=1",
    "v4":"https://www.nitrc.org/frs/download.php/12261/WHS_SD_rat_atlas_v4.label//?i_agree=1&download_now=1",
    "v3.01":"https://www.nitrc.org/frs/download.php/12258/WHS_SD_rat_atlas_v3.01.label//?i_agree=1&download_now=1",
    "v3":"https://www.nitrc.org/frs/download.php/11404/WHS_SD_rat_atlas_v3.label//?i_agree=1&download_now=1",
    "v2":"https://www.nitrc.org/frs/download.php/9439/WHS_SD_rat_atlas_v2.label//?i_agree=1&download_now=1",
    "v1.01":"https://www.nitrc.org/frs/download.php/9436/WHS_SD_rat_atlas_v1.label//?i_agree=1&download_now=1"
    }
annotation_urls = {
    "v4.01":"https://www.nitrc.org/frs/download.php/13398/WHS_SD_rat_atlas_v4.01.nii.gz//?i_agree=1&download_now=1",
    "v4":"https://www.nitrc.org/frs/download.php/12260/WHS_SD_rat_atlas_v4.nii.gz//?i_agree=1&download_now=1",
    "v3.01":"https://www.nitrc.org/frs/download.php/12257/WHS_SD_rat_atlas_v3.01.nii.gz//?i_agree=1&download_now=1",
    "v3":"https://www.nitrc.org/frs/download.php/11403/WHS_SD_rat_atlas_v3.nii.gz//?i_agree=1&download_now=1",
    "v2":"https://www.nitrc.org/frs/download.php/9438/WHS_SD_rat_atlas_v2.nii.gz//?i_agree=1&download_now=1",
    "v1.01":"https://www.nitrc.org/frs/download.php/9435/WHS_SD_rat_atlas_v1.01.nii.gz//?i_agree=1&download_now=1"
    }

data_dir = "../demo_data/"
for key in label_urls.keys():
    l_url = label_urls[key]
    a_url = annotation_urls[key]
    download_to_file(l_url, f"{data_dir}/{key}_label.label")
    reformat_WHS_label(f"{data_dir}/{key}_label.label", f"{data_dir}/waxholm_{key}_label.csv")
    download_to_file(a_url,  f"{data_dir}/{key}_anno.nii.gz")
    volume = nib.load(f"{data_dir}/{key}_anno.nii.gz")
    data = np.asarray(volume.dataobj)
    nrrd.write(f"{data_dir}/waxholm_{key}.nrrd", data)





import os, sys, glob, shutil, kaggle
import urllib.request as r
sys.path.append("./src")

def create_data(save_dir, data_name = "cells"):

    data_names = ["drone", "flood", "cells"]
    assert data_name in data_names, f"Please choose a proper dataset name from the list: {data_names}."

    if data_name == "drone": url = "kaggle datasets download -d killa92/drone-images-semantic-segmentation"
    elif data_name == "flood": url = "kaggle datasets download -d killa92/flood-image-segmentation"
    elif data_name == "cells": url = "kaggle datasets download -d killa92/medical-cells-image-segmentation"

    # Download from the checkpoint path
    if os.path.isfile(f"{save_dir}/{data_name}.csv") or os.path.isdir(f"{save_dir}/{data_name}"): print(f"The selected data is already donwloaded. Please check {save_dir}/{data_name} directory."); pass

    # If the checkpoint does not exist
    else:
        ds_name = url.split("/")[-1]
        print(f"{data_name} dataset is being downloaded...")
        # Download the dataset
        os.system(f"{url} -p {save_dir}")
        shutil.unpack_archive(f"{save_dir}/{ds_name}.zip", f"{save_dir}")
        os.remove(f"{save_dir}/{ds_name}.zip")
        # os.rename(f"{save_dir}/{ds_name}", f"{save_dir}/{data_name}")
        print(f"The selected dataset is downloaded and saved to {save_dir}/{data_name} directory!")

# create_data(save_dir = "datasets", data_name = "cells")
# create_data(save_dir = "datasets", data_name = "flood")
# create_data(save_dir = "datasets", data_name = "drone")
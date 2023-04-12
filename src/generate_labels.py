from datasets import S2Data, S2RawData, S2CloudlessData, S2ExolabData
import numpy as np
from tqdm import tqdm
from additional_data import load_additional_data
import matplotlib.pyplot as plt

res = 60


def compute_CE_labels(cloudless_img: np.ndarray, exolabs_img: np.ndarray):
    """
    Compute labels based on s2cloudless/exolabs classification.
    Clouds (with prob > 0.5) cover everything else.
    """
    label = np.concatenate([cloudless_img, exolabs_img], axis=0)
    label = np.argmax(label, axis=0)
    label[cloudless_img.squeeze() > 0.5] = 0
    return label.astype(np.uint8)


def compute_CE_water_labels(cloudless_img: np.ndarray, exolabs_img: np.ndarray, water_img: np.ndarray):
    """
    Computes images based on s2cloudless/exolabs classification adding the water data.

    Snow covers water, clouds (with prob > 0.5) cover everything else.
    """
    label = np.concatenate([cloudless_img, exolabs_img], axis=0)
    label = np.argmax(label, axis=0)
    snow = label == 2
    water = water_img == 1
    label[water] = 3
    label[snow] = 2
    label[cloudless_img.squeeze() > 0.5] = 0
    return label.astype(np.uint8)


additional_data = load_additional_data(res=res)

raw_data = S2RawData()
# raw_dataset = raw_data._load_dataset_from_images(save_arrays=False, return_arrays=True, resolutions=[res], ids_to_load=["20210526"])

cloudless_data = S2CloudlessData()
cloudless_dataset = cloudless_data._load_dataset_from_images(save_arrays=False,
                                                             return_arrays=True,
                                                             ids_to_load=list(raw_data.images.keys()),
                                                             resolutions=[res])

exolab_data = S2ExolabData()
exolab_dataset = exolab_data._load_dataset_from_images(save_arrays=False,
                                                       return_arrays=True,
                                                       ids_to_load=list(raw_data.images.keys()),
                                                       resolutions=[res],
                                                       convert_channels=True)

labels = {}
labels[res] = {}
print(f"Computing labels for {res}m resolution...")
for id in tqdm(cloudless_dataset[res]):
    cloudless_img = cloudless_dataset[res][id]
    cloudless_img = cloudless_data.preprocess(cloudless_img).squeeze(axis=0)
    exolab_img = exolab_dataset[res][id]
    exolab_img = exolab_data.preprocess(exolab_img).squeeze(axis=0)

    label = compute_CE_water_labels(cloudless_img, exolab_img, additional_data["surface_water"])
    labels[res][id] = label
print("saving npz...")
np.savez(f"data/labels/cloudless_exolabs_water/{res}m.npz", **labels[res])

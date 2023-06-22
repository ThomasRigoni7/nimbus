from datasets import S2Data, S2RawData, S2CloudlessData, S2ExolabData
import numpy as np
from tqdm import tqdm
from additional_data import load_additional_data
from argparse import ArgumentParser
from pathlib import Path

def compute_CE_water_label(cloudless_img: np.ndarray, exolab_img: np.ndarray, water_img: np.ndarray):
    nodata = exolab_img[0] == 0
    snow = exolab_img[1] == 2
    no_snow = exolab_img[1] == 1
    dark_regions = exolab_img[0] == 2
    no_snow = np.logical_or(no_snow, dark_regions)
    clouds = cloudless_img.squeeze() > 0.5
    water = water_img == 1
    label = np.zeros_like(nodata, dtype=np.uint8)
    label[no_snow] = 1
    label[water] = 3
    label[snow] = 2
    label[clouds] = 0
    return label

def compute_E_water_label(exolab_img: np.ndarray, water_img: np.ndarray):
    """
    Computes images based on exolabs classification adding the water data.

    Snow covers water, clouds cover everything else.
    """
    label = np.ones((exolab_img.shape[-2], exolab_img.shape[-1]), dtype=np.uint8)
    clouds = exolab_img[0] == 3
    snow = exolab_img[0] == 4
    water = water_img == 1
    label[water] = 3
    label[snow] = 2
    label[clouds] = 0
    return label

def mask_label(label, exolab_img):
    label[exolab_img[0] == 0] = 255
    return label

def generate_CE_labels(exolabs_folder: str|Path, cloudless_folder:str|Path, tile:str, res:int, masked:bool):
    additional_data = load_additional_data(res=res, tile=tile)
    exolab_data = S2ExolabData(data_dir=exolabs_folder)
    exolab_dataset = exolab_data._load_dataset_from_images(save_arrays=False,
                                                        return_arrays=True,
                                                        resolutions=[res])

    cloudless_data = S2CloudlessData(data_dir=cloudless_folder)
    cloudless_dataset = cloudless_data._load_dataset_from_images(save_arrays=False,
                                                                 return_arrays=True,
                                                                 resolutions=[res])

    labels = {}
    print(f"Computing labels for {res}m resolution...")
    for id in tqdm(exolab_dataset[res]):
        cloudless_img = cloudless_dataset[res][id]
        cloudless_img = cloudless_data.preprocess(cloudless_img).squeeze(axis=0)
        exolab_img = exolab_dataset[res][id]

        label = compute_CE_water_label(cloudless_img, exolab_img, additional_data["surface_water"])
        if masked:
            label = mask_label(label, exolab_img)
        labels[id] = label
    return labels
    
def generate_exolab_labels(exolabs_folder: str|Path, tile:str, res:int, masked:bool):
    additional_data = load_additional_data(res=res, tile=tile)
    exolab_data = S2ExolabData(data_dir=exolabs_folder)
    exolab_dataset = exolab_data._load_dataset_from_images(save_arrays=False,
                                                        return_arrays=True,
                                                        resolutions=[res])

    labels = {}
    print(f"Computing labels for {res}m resolution...")
    for id in tqdm(exolab_dataset[res]):
        exolab_img = exolab_dataset[res][id]
        label = compute_E_water_label(exolab_img, additional_data["surface_water"])
        if masked:
            label = mask_label(label, exolab_img)
        labels[id] = label
    
    return labels


if __name__ == "__main__":
    parser = ArgumentParser(description="Generates labels in numpy .npz format for a set of images from the exolabs classification.")
    parser.add_argument("exolabs_folder", type=str, help="folder containing exolabs image folders.")
    parser.add_argument("dst_folder", type=str, help="folder into which to create the label file.")
    parser.add_argument("-tile", type=str, help="name of the tile, used to load the appropriate additional data.", default="32TNS")
    parser.add_argument("-cloudless_folder", type=str, help="""folder containing the S2cloudless images, 
                        if specified then the algorithm will ignore the exolabs cloud classification and use S2cloudless instead.""")
    parser.add_argument("-res", type=int, help="resolution", choices=[10, 20, 60], default=10)
    parser.add_argument("-masked", action="store_true", help="if specified, mask the labels with the exolabs nodata mask. otherwie ignore it.")

    args = parser.parse_args()

    exolabs_folder = Path(args.exolabs_folder)
    dst_folder = Path(args.dst_folder)
    res = args.res
    tile = args.tile
    masked = args.masked
    
    if args.cloudless_folder is None:
        labels = generate_exolab_labels(exolabs_folder, tile, res, masked)
    else:
        s2cloudless_folder = Path(args.cloudless_folder)
        labels = generate_CE_labels(exolabs_folder, s2cloudless_folder, tile, res, masked)

    print("saving npz...")
    path = dst_folder / f"{res}m.npz"
    path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(path, **labels)
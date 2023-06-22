import numpy as np
import torch
from pathlib import Path
from segment import save_prediction_geotif


def save_label_tif(tile: str, image_dir:str|Path, out_path: str|Path):
    labels = np.load(f"data/labels/cloudless_exolabs_water/{tile}/10m.npz")

    l = labels[image_dir.name[:8]]
    label = torch.from_numpy(l)
    save_prediction_geotif(label, image_dir, out_path)
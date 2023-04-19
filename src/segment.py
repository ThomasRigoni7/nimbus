import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from segnet import SegNetLite
from datasets import S2RawData, S2Data, ActiveLearningDataset
import pytorch_lightning as pl
from main import load_models
import rasterio
import gdal
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
OUT_CLASSES = 4

# add command line options
def parse_arguments():
    parser = argparse.ArgumentParser("Script used to segment a full raw image from L1C satellite images.")
    parser.add_argument("-img_dir", type=str, default="data/raw/raw_processed/20210710T101559")
    parser.add_argument("-out", type=str, default="segmentation.tif")
    parser.add_argument("-model", type=str, default="checkpoints/epoch=59-step=1800.ckpt")
    parser.add_argument("--resolution", type=int, choices=[10, 20, 60], default=10)
    parser.add_argument("--cut_dim", type=int, default=512)
    parser.add_argument("--cut_offset", type=int, default=32)
    return parser.parse_args()

def load_image(image_dir: str | Path, resolution: int) -> torch.Tensor:
    raw_data = S2RawData()
    full_image = raw_data._load_full_image(image_dir, pixel_resolution=resolution)
    full_image = raw_data.preprocess(full_image).squeeze()
    return torch.from_numpy(full_image)

def predict(model: nn.Module, image: torch.Tensor) -> dict[str, torch.Tensor]:
    model = model.to(DEVICE)
    model.return_predict_probabilities = True
    image_dict = {"image": image}
    ALdataset = ActiveLearningDataset(image_dict, None, 512, training_ids=list(image_dict.keys()), additional_layers=["altitude", "treecover"], test_labels=None)
    _, dataset, _ = ALdataset.get_datasets(0)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
    trainer = pl.Trainer(devices=1, accelerator="gpu", logger=None)
    predictions = trainer.predict(model, loader, return_predictions=True)
    ret = {}
    for pred in predictions:
        ret.update(pred)
    return ret
    
def merge_predictions(predictions: dict[str, torch.Tensor], original_size: int):
    full_image = torch.zeros((OUT_CLASSES, original_size, original_size))
    image_counts = torch.zeros((OUT_CLASSES, original_size, original_size))
    for id, pred in predictions.items():
        h, w = pred.shape[-2], pred.shape[-1]
        x_y = id.split("/")[-1]
        x, y = x_y.split("_")
        x, y = int(x), int(y)
        print(id)
        print(x, y)
        print()
        full_image[:, x:x+h, y:y+w] += pred
        image_counts[:, x:x+h, y:y+w] += 1
    segmentation_mask = torch.argmax(full_image / image_counts, dim=0)
    plt.imshow(segmentation_mask.numpy())
    plt.show()
    return segmentation_mask.to(dtype=torch.uint8)

def save_prediction(segmentation_mask: torch.Tensor, img_dir: Path | str, path_to_save: Path | str, res: int = 10):
    """
    Would be good to save as geotif so that it is possible to look at it with Qgis.
    """
    single_band_image = list(Path(img_dir).glob("*"))[0]
    print("Saving prediction to disk!")
    ds = gdal.Open(str(single_band_image))
    ds = gdal.Translate("/vsimem/in_memory_output.tif", ds, xRes = res, yRes=res)
    in_band = ds.GetRasterBand(1)
    arr = in_band.ReadAsArray()
    [rows, cols] = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path_to_save, cols, rows, 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    out_band = outdata.GetRasterBand(1)
    # set color interpretation
    colors = gdal.ColorTable()
    colors.SetColorEntry(0, (255, 0, 0))
    colors.SetColorEntry(1, (112, 153, 89))
    colors.SetColorEntry(2, (115, 191, 250))
    colors.SetColorEntry(3, (242, 238, 162))
    out_band.SetRasterColorTable(colors)
    out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    out_band.WriteArray(segmentation_mask.numpy())
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    in_band=None
    ds=None

def show_prediction(full_image: torch.Tensor, segmentation_mask: torch.Tensor):
    pass

if __name__ == "__main__":
    args = parse_arguments()
    full_image = load_image(args.img_dir, args.resolution)
    initial_dim = full_image.shape[-1]
    model = load_models(args.model)
    preds = predict(model, full_image)
    final_segmentation = merge_predictions(preds, initial_dim)
    # show_prediction(full_image, final_segmentation)
    save_prediction(final_segmentation, args.img_dir, args.out, res=args.resolution)


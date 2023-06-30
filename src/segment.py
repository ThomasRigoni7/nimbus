import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from datasets import S2RawData, ActiveLearningDataset
import pytorch_lightning as pl
from main import load_models
import gdal
import math
import matplotlib.pyplot as plt
from additional_data import load_additional_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
OUT_CLASSES = 4

# add command line options
def parse_arguments():
    parser = argparse.ArgumentParser("Script used to segment a full raw image from L1C satellite images.")
    parser.add_argument("-img_dir", type=str, default="data/raw/raw_processed/20210126T102311")
    parser.add_argument("-out", type=str, default="segmentation.tif")
    parser.add_argument("-model", type=str, default="checkpoints/AL/final/fpn_5.ckpt")
    parser.add_argument("-tile", type=str, help="tile where the images belong, used to load the DEM and tree cover additional data.", default="32TNS")
    parser.add_argument("--resolution", type=int, choices=[10, 20, 60], default=10)
    parser.add_argument("--cut_dim", type=int, default=512)
    parser.add_argument("--cut_offset", type=int, default=256)
    parser.add_argument("--log_normalize", help="If set, log normalize the images", action="store_true")
    return parser.parse_args()

def load_image(image_dir: str | Path, resolution: int, log_normalize:bool=False) -> torch.Tensor:
    raw_data = S2RawData(log_normalize=log_normalize)
    image_dir = Path(image_dir)
    if image_dir.is_dir():
        full_image = raw_data._load_full_image(image_dir, pixel_resolution=resolution)
    elif image_dir.is_file():
        full_image = raw_data._load_full_image_tif(image_dir, pixel_resolution=resolution)
    full_image = raw_data.preprocess(full_image).squeeze()
    return torch.from_numpy(full_image)

def predict(model: nn.Module, image: torch.Tensor, cut_dim=512, cut_offset=32) -> dict[str, torch.Tensor]:
    model = model.to(DEVICE)
    model.return_predict_probabilities = True
    image_dict = {"image": image}
    ALdataset = ActiveLearningDataset(image_dict, None, cut_dim, training_ids=list(image_dict.keys()), input_bands="all", additional_layers=["altitude", "treecover"], test_labels={}, image_overlap=cut_offset)
    _, dataset, _, _, _= ALdataset.get_datasets([], [], [], 0, compute_class_weights=False)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
    trainer = pl.Trainer(devices=[2], accelerator="gpu", logger=None)
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
        full_image[:, x:x+h, y:y+w] += pred
        image_counts[:, x:x+h, y:y+w] += 1
    segmentation_mask = torch.argmax(full_image / image_counts, dim=0)
    # plt.imshow(segmentation_mask.numpy())
    # plt.show()
    return segmentation_mask.to(dtype=torch.uint8)

def save_prediction_geotif(segmentation_mask: torch.Tensor, original_img_dir_or_path: Path | str, path_to_save: Path | str, res: int = 10):
    """
    Saves a prediction as a geoTiff image.
    """
    if Path(original_img_dir_or_path).is_dir():
        single_band_image = list(Path(original_img_dir_or_path).glob("*"))[0]
    else:
        single_band_image = Path(original_img_dir_or_path)
    print("Saving prediction to disk!")
    ds = gdal.Open(str(single_band_image))
    ds = gdal.Translate("/vsimem/in_memory_output.tif", ds, xRes = res, yRes=res)
    in_band = ds.GetRasterBand(1)
    arr = in_band.ReadAsArray()
    [rows, cols] = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(str(path_to_save), cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
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
    out_band.SetNoDataValue(255)


    # account for the fact that the tiff images are not squares and don't start at 0,0 but at 1,1
    mask = segmentation_mask.numpy()
    if mask.shape != (rows, cols):
        base = np.ones((rows, cols), dtype=np.uint8) * 255
        base[1:mask.shape[0]+1, 1:mask.shape[1]+1] = mask
        mask = base

    out_band.WriteArray(mask)
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    in_band=None
    ds=None

def predict_smooth(model, image, res, cut_dim, tile="32TNS"):
    print("Predicting smooth!")
    from smooth_tiled_predictions import predict_img_with_smooth_windowing
    model.eval()
    model = model.to("cuda:0")
    batch_size = 64
    def pred_func(cut_batch_np):
        preds = []
        cut_batch = torch.from_numpy(cut_batch_np.transpose(0, 3, 1, 2))
        for b in range(math.ceil(len(cut_batch) / batch_size)):
            batch = cut_batch[b*batch_size:(b+1)*batch_size]
            batch = batch.to("cuda:0")
            pred = model.model(batch).detach().cpu()
            preds.append(pred)
        preds = torch.cat(preds, dim=0)
        return preds.numpy().transpose(0, 2, 3, 1)
    
    additional_layers = ["altitude", "treecover"]
    additional_data = load_additional_data(res = res, tile=tile)
    layers = []
    for layer_name in additional_layers:
            layer = additional_data[layer_name]
            layers.append(layer[None, :])
        
    image = S2RawData.filter_bands(image.numpy(), ["B1", "B2", "B3", "B4","B5","B6","B7","B8", "B8A", "B9","B10","B11","B12"])
    nodata_mask = np.any(image == 0, axis=0)
    image = np.concatenate([image] + layers, axis=0)

    image = image.transpose(1, 2, 0)
    pred = predict_img_with_smooth_windowing(image, cut_dim, 2, OUT_CLASSES, pred_func)
    pred = torch.from_numpy(pred.transpose(2, 0, 1))
    pred = torch.argmax(pred, dim=0).to(dtype=torch.uint8)
    pred[torch.from_numpy(nodata_mask)] = 255
    return pred

def predict_and_save_segmentation(  img_dirs: list[str] | list[Path], 
                                    model_path: str | Path, 
                                    outs:list[str] | list[Path], 
                                    res: int, 
                                    cut_dim: int, 
                                    log_normalize:bool=False, 
                                    tile:str="32TNS"):
    """
    runs inference on the model with smooth predictions and saves the generated segmentation mask(s).
    Arguments:
     - img_dirs: list[str] | list[Path] contains the paths of the directories that contain the images.
     - model_path: str | Path contains the path to the model checkpoint.
     - outs: list[str] | list[Path] contains the paths of the output files in which to place the segmentation masks.
     - res: int input/output resolution.
     - cut_dim: int size of the squares in which the image is cut to run inference.
     - log_normalize: bool if set, use log-normalization to process the images.
    """
    model = load_models(None, model_path)
    for img_dir, out in zip(img_dirs, outs):
        print("Processing", img_dir)
        full_image = load_image(img_dir, res, log_normalize)
        final_segmentation = predict_smooth(model, full_image, res, cut_dim, tile)
        save_prediction_geotif(final_segmentation, img_dir, out, res)

def predict_all_dataset(model_path, data_path, out_folder, res, cut_dim, tile:str="32TNS"):
    img_dirs = list(Path(data_path).glob("*"))
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    outs = [(out_folder / d.name).with_suffix(".tif") for d in img_dirs]
    predict_and_save_segmentation(img_dirs, model_path, outs, res, cut_dim, tile=tile)

if __name__ == "__main__":
    # args = parse_arguments()
    # predict_and_save_segmentation([args.img_dir], args.model, [args.out], args.resolution, args.cut_dim, args.log_normalize, args.tile)
    predict_all_dataset("checkpoints/AL/final/fpn_5.ckpt", "data/new_data/32TNS/new", "predictions/fpn_5/32TNS_new", 10, 512)

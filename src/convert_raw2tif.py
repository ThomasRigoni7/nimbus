# script used to generate tif rgb or false color images from raw multi-file single band data.

from pathlib import Path
import gdal
import numpy as np
import argparse
from datasets import S2RawData

def save_image_geotif(image: np.ndarray, img_dir: Path | str, path_to_save: Path | str, compress:bool, res: int = 10):
    """
    Image has shape [3, h, w]
    """
    if len(image) != 3:
        raise ValueError("Length of image is not 3: cannot use it as RGB.")
    single_band_image = list(Path(img_dir).glob("*"))[0]
    print("Saving prediction to disk!")
    ds = gdal.Open(str(single_band_image))
    ds = gdal.Translate("/vsimem/in_memory_output.tif", ds, xRes = res, yRes=res)
    in_band = ds.GetRasterBand(1)
    arr = in_band.ReadAsArray()
    [rows, cols] = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    if compress:
        outdata = driver.Create(str(path_to_save), cols, rows, 3, gdal.GDT_Byte, options=['COMPRESS=JPEG'])
    else:
        outdata = driver.Create(str(path_to_save), cols, rows, 3, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    for band_number, interpretation, band_image in zip(range(1, 4), [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand], image):
        out_band = outdata.GetRasterBand(band_number)
        if compress:
            out_band.WriteArray(np.clip(band_image // 60, 0, 255).astype(np.int8))
        else:
            out_band.WriteArray(band_image)
        out_band.SetRasterColorInterpretation(interpretation)
    
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    in_band=None
    ds=None


def get_bands(full_image, bands: list[int]|str):
    if bands == "rgb":
        bands = ["B4", "B3", "B2"]
    elif bands == "fci":
        bands = ["B12", "B8", "B4"]
    if isinstance(bands, str) and bands != "rgb" and bands != "fci":
        raise ValueError(f"Unrecognized bands: {bands}.")
    
    return S2RawData.filter_bands(full_image, bands)
    
def convert_raw2tif(image_dir: str|Path, out: str|Path, bands:str, compress:bool):
    raw_data = S2RawData()
    print("Loading image...")
    full_image = raw_data._load_full_image(image_dir, pixel_resolution=10)
    image_bands = get_bands(full_image, bands)
    save_image_geotif(image_bands, image_dir, out, compress)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="path of the directory containing the image bands.")
    parser.add_argument("out", type=str, help="name of the output file to create.")
    parser.add_argument("bands", help="bands to use: either 'rgb' or 'fci'", choices=["rgb", "fci"], default="fci")
    parser.add_argument("-compress", help="if specified, compress the image with lossy JPEG format.", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    convert_raw2tif(args.image_dir, args.out, args.bands, args.compress)
    
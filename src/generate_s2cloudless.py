# script to generate s2cloudless cloud probabilities. It does not run with the provided conda environment, due to a package version issue.

from argparse import ArgumentParser
from pathlib import Path

from datasets import S2RawData
from s2cloudless import S2PixelCloudDetector
from osgeo import gdal
import numpy as np

parser = ArgumentParser(description="Generates the S2Cloudless cloud probabilities from Sentinel2 L1C image files.")
parser.add_argument("src_folder", type=str, help="folder containing the L1C band files.")
parser.add_argument("dst_folder", type=str, help="folder which will contain the output cloud probability geotiff files.")

args = parser.parse_args()

original_folder = Path(args.src_folder)
results_folder = Path(args.dst_folder)
results_folder.mkdir(exist_ok=True, parents=True)
cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=False)

raw_data = S2RawData(data_dir=original_folder)
for image_folder in original_folder.glob("*"):
    image = raw_data._load_full_image(image_folder)
    image = raw_data.filter_bands(image, ["B1", "B2", "B4", "B5", "B8", "B8A", "B9", "B10", "B11", "B12"])
    cloud_mask = cloud_detector.get_cloud_probability_maps(image.transpose(1, 2, 0)[None, ...]).squeeze()
    single_band_image = list(Path(image_folder).glob("*"))[0]
    ds = gdal.Open(str(single_band_image))
    in_band = ds.GetRasterBand(1)
    arr = in_band.ReadAsArray()
    [rows, cols] = arr.shape

    path_to_save = results_folder / image_folder.with_suffix(".tif").name
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(str(path_to_save), cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    out_band = outdata.GetRasterBand(1)

    out_band.WriteArray((cloud_mask* 100).astype(np.uint8))
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    ds = None
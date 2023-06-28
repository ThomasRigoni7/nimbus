# mask net predictions with the labels nodata mask

import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import S2RawData
from osgeo import gdal

tiles = ["32TNS", "13TDE", "07VEH", "32VMP"]

raw_location = {
    "32TNS": Path("data/raw/raw_processed"),
    "13TDE": Path("data/new_data/13TDE"),
    "07VEH": Path("data/new_data/07VEH"),
    "32VMP": Path("data/new_data/32VMP")
}


for tile in tiles:
    print(tile)
    predictions_path = Path(f"predictions/fpn_5/{tile}/")

    raw_data = S2RawData(data_dir=raw_location[tile])

    for p in tqdm(list(predictions_path.glob("*.tif"))):
        original_image_path = (raw_location[tile] / p.with_suffix("").name)
        ds = gdal.Open(str(p), gdal.GA_Update)
        pred_band = ds.GetRasterBand(1)
        pred_band.SetNoDataValue(255)
        pred = pred_band.ReadAsArray()

        image = raw_data._load_full_image(original_image_path)
        image = raw_data.filter_bands(image, ["B1", "B2", "B3", "B4","B5","B6","B7","B8", "B8A", "B9","B10","B11","B12"])
        
        mask = np.any(image == 0, axis=0)
        pred[mask] = 255

        pred_band.WriteArray(pred)
        ds.FlushCache() ##saves to disk!!
        ds = None
        in_band=None
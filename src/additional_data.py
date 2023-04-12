"""
Script to load and convert the additional data into new files with 10m resolution and cut the padding pixels
"""

from pathlib import Path
import numpy as np
import gdal

additional_data_filenames = {
    "lakes": "32TNS_10m_COP_Lakes.tif",
    "glaciers": "32TNS_30m_Glacier_RGIv6.tif",
    "surface_water": "32TNS_30m_JRC_surfaceWater.tif",
    "altitude": "DEM_32TNS_10m_epsg32632.tif",
    "landcover": "32TNS_30m_landcover_CORINE_2018.tif",
    "treecover": "32TNS_30m_treeCanopyCover.tif"
}

def convert_additional_data(additional_data_dir: str | Path = Path("data/auxilliary_data"), save_files=True) -> dict[str, np.ndarray]:
    """
    loads, converts and saves additional data like lakes, glaciers, surface water and altitude.
    Returns all the images in shape [height, width].
    """
    additional_data_dir = Path(additional_data_dir)
    new_dir = additional_data_dir / "processed"
    new_dir.mkdir(exist_ok=True)
    ret = {}
    for name, filename in additional_data_filenames.items():
        img_size = 10980
        if "30m" in filename:
            img_size = img_size // 3
        ds = gdal.Open(str(additional_data_dir / filename))
        ds = gdal.Translate("/vsimem/in_memory_output.tif", ds, srcWin = [1, 1, img_size, img_size], xRes = 10, yRes=10)
        ret[name] = ds.ReadAsArray()
        print(np.count_nonzero(ret[name] == 1))
        ds = None
    return ret

def load_additional_data(processed_dir: str | Path = Path("data/auxilliary_data/processed/"), res=10) -> dict[str, np.ndarray]:
    """
    loads additional data like lakes, glaciers, surface water and altitude.
    Returns all the images in shape [height, width].
    """
    processed_dir = Path(processed_dir)
    ret = {}
    for name, filename in additional_data_filenames.items():
        ds = gdal.Open(str(processed_dir /f"32TNS_10m_{name}.tif"))
        if res != 10:
            ds = gdal.Translate("/vsimem/in_memory_output.tif", ds, xRes = res, yRes=res)
        ret[name] = ds.ReadAsArray()
        ds = None
    return ret

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    additional_data = convert_additional_data()
    # data = load_additional_data()
    # for name, img in data.items():
    #     print(img.shape)
    #     plt.imshow(img)
    #     plt.title(name)
    #     plt.show()
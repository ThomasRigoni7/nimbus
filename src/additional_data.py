"""
Script to load and convert the additional data into new files with 10m resolution and cut the padding pixels
"""

from pathlib import Path
import numpy as np
import gdal

gdal.UseExceptions()

additional_data_filenames = {
    "lakes": "{}_10m_COP_Lakes.tif",
    "glaciers": "{}_30m_Glacier_RGIv6.tif",
    "surface_water": "{}_30m_JRC_surfaceWater.tif",
    # "altitude": "DEM_{}_10m_epsg32632.tif", -> this was only available for 32TNS in 10m, all others are 30m resolution
    "altitude": "{}_30m_DEM_AW3D30.tif",
    "landcover": "{}_30m_landcover_CORINE_2018.tif",
    "treecover": "{}_30m_treeCanopyCover.tif"
}
additional_data_normalization_values = {
    "lakes": 1,
    "glaciers": 1,
    "surface_water": 1,
    "altitude": 5000,
    "landcover": 255,
    "treecover": 100
}

def convert_additional_data(additional_data_dir: str | Path = Path("data/auxilliary_data"), tile:str = "32TNS") -> dict[str, np.ndarray]:
    """
    loads, converts and saves additional data like lakes, glaciers, surface water and altitude.
    Returns all the images in shape [height, width].
    """
    additional_data_dir = Path(additional_data_dir)
    processed_dir = additional_data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    tile_dir = additional_data_dir / tile
    ret = {}
    for name, filename in additional_data_filenames.items():
        file_path = tile_dir / filename.format(tile)
        if not file_path.exists():
            continue
        print("processing:", file_path)
        img_size = 10980
        if "30m" in filename:
            img_size = img_size // 3
        ds = gdal.Open(str(file_path))
        new_path = processed_dir/ f"{tile}_10m_{name}.tif"
        print("saving in:", new_path)
        out_ds = gdal.Translate(str(new_path), ds, srcWin = [1, 1, img_size, img_size], xRes = 10, yRes=10)
        ret[name] = out_ds.ReadAsArray()
        ds = None
        out_ds = None
    return ret

def normalize_additional_data(data: dict[str, np.ndarray]):
    """
    Brings all the values of additional data images to the range [0, 1]
    """
    ret = {}

    for id, img in data.items():
        ret[id] = img.astype(np.float32) / additional_data_normalization_values[id]

    return ret

def load_additional_data(processed_dir: str | Path = Path("data/auxilliary_data/processed/"), res=10, normalize:bool=True, tile: str="32TNS") -> dict[str, np.ndarray]:
    """
    loads additional data like lakes, glaciers, surface water and altitude.
    Returns all the images in shape [height, width].
    """
    print(f"Loading additional data for tile {tile}.")
    processed_dir = Path(processed_dir)
    ret = {}
    for name, filename in additional_data_filenames.items():
        file = (processed_dir /f"{tile}_10m_{name}.tif")
        if not file.exists():
            continue
        ds = gdal.Open(str(processed_dir /f"{tile}_10m_{name}.tif"))
        if res != 10:
            ds = gdal.Translate("/vsimem/in_memory_output.tif", ds, xRes = res, yRes=res)
        ret[name] = ds.ReadAsArray()
        ds = None
    if normalize:
        ret = normalize_additional_data(ret)
    return ret

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    additional_data = convert_additional_data(tile="32VMP")
    # data = load_additional_data()
    # for name, img in data.items():
    #     print(img.shape)
    #     plt.imshow(img)
    #     plt.title(name)
    #     plt.show()
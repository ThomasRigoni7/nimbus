# script used to cut windows of tif images into png to generate images for the report.


from osgeo import gdal
from pathlib import Path
from tqdm import tqdm

folder = Path("prova/")
windows = {
    # 32TNS
    "20211227T102339": [1750, 1750, 2048, 2048],
    "20210111T102309": [7500,  3500, 2048, 2048],
    # 07VEH
    "20210206T204629": [1600, 7400, 2048, 2048],
    "20210209T205619": [4800, 6300, 2048, 2048],
    # 13TDE
    "20211008T175241": [7100, 6000, 2048, 2048],
    "20211222T175739": [1800, 4000, 2048, 2048],
    "20210426T174859": [1100, 8700, 2048, 2048],
    # 32VMP
    "20210307T110819": [4100, 2100, 2048, 2048],
    "20210503T105619": [4000, 2500, 2048, 2048],
    "20211126T105309": [7200, 6200, 2048, 2048],
    "20211206T105329": [2900,  100, 2048, 2048]
    }

def convert_tif2png(path: Path, window: list[int] | None = None):
    ds = gdal.Open(str(path))

    png_driver = gdal.GetDriverByName("PNG")
    
    if window is not None:
        tmp_ds = png_driver.CreateCopy("/vsimem/tmp.png", ds)
        png_ds = gdal.Translate(str(path.with_suffix(".png")), tmp_ds, srcWin = window)
        tmp_ds = None
    else:
        png_ds = png_driver.CreateCopy(str(path.with_suffix(".png")), ds)
    
    png_ds.FlushCache()

    ds = None
    png_ds = None
    path.with_suffix(".png.aux.xml").unlink()

for file in tqdm(list(folder.glob("**/*.tif"))):
    convert_tif2png(file, window=[10,  7900, 3000, 3000])
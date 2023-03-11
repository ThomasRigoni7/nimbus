from s2data import S2Data
from pathlib import Path
import rasterio
import numpy as np

class S2RawData(S2Data):
    def __init__(self, dataset_dir = Path("data/raw/"), data_dir = Path("data/raw/raw_processed/")):
        super().__init__(dataset_dir)
        image_paths = list(data_dir.glob("*"))
        for image_path in image_paths:
            date = image_path.name[:8]
            self.images[date] = image_path

    def _load_full_image(self, image: str, pixel_resolution: int = 10) -> np.ndarray:
        """
        Loads the full image (all bands) in the specified resolution, resizing the bands at different resolution.
        
        Band order is:
         - 0: Cloud probability
         - 1: Snow probability
         - 2: B01
         - 3: B02
         - 4: B03
         - 5: B04
         - 6: B05
         - 7: B06
         - 8: B07
         - 9: B08
         - 10: B09
         - 11: B10
         - 12: B11
         - 13: B12
         - 14: B8A
        """
        bands = list(self.images[image].glob("*.jp2"))
        bands.sort()
        l = []
        for b in bands:
            with rasterio.open(b, 'r') as f:
                img = f.read(1)
                # check if interpolation is necessary
                img = self._convert_to_resolution(img, pixel_resolution)
                l.append(img)
        res = np.stack(l)
        return res


def _test():
    s2data = S2RawData()
    s2data.load_arrays(resolution=60)

if __name__ == "__main__":
    _test()
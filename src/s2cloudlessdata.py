from s2data import S2Data
from pathlib import Path
import rasterio
from rasterio.windows import Window

class S2CloudlessData(S2Data):
    def __init__(self, dataset_dir = Path("data/S2cloudless/"), data_dir = Path("data/S2cloudless/data/")):
        super().__init__(dataset_dir)
        image_paths = list(data_dir.glob("*.tif"))
        for image_path in image_paths:
            date = image_path.name[13:21]
            self.images[date] = image_path

    def _load_full_image(self, image: str, pixel_resolution: int = 10):
        """
        Loads the full image in the specified resolution, resizing the bands at different resolution.
        S2 cloudless data has some buffer pixels at (1 pixel up and left, 2 right, more in the bottom)
        """
        with rasterio.open(self.images[image], "r") as f:
            img = f.read(window=Window(1, 1, self.NATIVE_DIM, self.NATIVE_DIM))
            img = self._convert_to_resolution(img, pixel_resolution)
            return img

def _test():
    data = S2CloudlessData()
    data._load_and_save_dataset(data.array_locations[10])
    data._create_smaller_resolutions()

if __name__ == "__main__":
    _test()
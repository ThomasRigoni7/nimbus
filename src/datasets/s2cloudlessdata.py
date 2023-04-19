from datasets.s2data import S2Data
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np

class S2CloudlessData(S2Data):
    def __init__(self, dataset_dir = Path("data/S2cloudless/"), data_dir = Path("data/S2cloudless/data/"), resolution: int = 10):
        super().__init__(dataset_dir, resolution)
        image_paths = list(data_dir.glob("*.tif"))
        for image_path in image_paths:
            date = image_path.name[13:21]
            self.images[date] = image_path

    def _load_full_image(self, image_path: Path | str, pixel_resolution: int = 10):
        """
        Loads the full image in the specified resolution, resizing the bands at different resolution.
        S2 cloudless data has some buffer pixels at (1 pixel up and left, 2 right, more in the bottom)
        """
        out_size = (self.NATIVE_RESOLUTION * self.NATIVE_DIM) // pixel_resolution
        with rasterio.open(str(image_path), "r") as f:
            img = f.read(window=Window(1, 1, self.NATIVE_DIM, self.NATIVE_DIM),
                         out_shape=(f.count, out_size, out_size),
                         resampling=Resampling.bilinear)
            return img

    def _normalize(self, imgs: np.ndarray):
        # Images in S2Cloudless are in range [0, 100]
        ret = (imgs.astype(np.float32) / 100).clip(0, 1)
        return ret

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        """
        Returns normalized images in range [0, 1] of shape [num_images, 1, height, width]
        """
        if imgs.ndim == 2:
            imgs = imgs[None, None:]
        elif imgs.ndim == 3:
            imgs = imgs[None, :]
        return self._normalize(imgs)


def _test():
    data = S2CloudlessData()
    array = data.load_array("20210101/0_0")
    print("Single array shape:", array.shape)
    arrays = data.load_arrays(ids_to_load=["20210101/0_0", "20210101/0_480"])
    print("Multiple arrays shape:", arrays.shape)


if __name__ == "__main__":
    _test()
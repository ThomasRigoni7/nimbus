from datasets import S2Data
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np


class S2ExolabData(S2Data):

    def __init__(self,
                 dataset_dir=Path("data/ExoLabs_classification_S2/"),
                 data_dir=Path("data/ExoLabs_classification_S2/data/"),
                 resolution: int = 10):
        super().__init__(dataset_dir, resolution)
        image_paths = list(data_dir.glob("*.tif"))
        for image_path in image_paths:
            date = image_path.name[9:19].replace("-", "")
            self.images[date] = image_path

    def _load_full_image(self, image: str, pixel_resolution: int = 10):
        """
        Loads the full image in the specified resolution, resizing the bands at different resolution.
        Exolabs data has some buffer pixels at (1 pixel up and left, 2 right, more in the bottom)
        """
        out_size = (self.NATIVE_RESOLUTION * self.NATIVE_DIM) // pixel_resolution
        with rasterio.open(self.images[image], "r") as f:
            img = f.read(window=Window(1, 1, self.NATIVE_DIM, self.NATIVE_DIM),
                         out_shape=(f.count, out_size, out_size),
                         resampling=Resampling.nearest)
            return img

    def _convert_channels(self, images: np.ndarray):
        """
        converts the images from categorical labels to multi-channel: 
        creates new channels with values of 0/1 where the appropriate label is met.

        returns an array with 2 channels: [no-snow, snow] and 4 dims in total: [images, channels, width, height]
        """
        no_snow = images[:, 1] == 1
        snow = images[:, 1] == 2
        return np.stack([no_snow, snow]).swapaxes(0, 1).astype(np.float32)

    def _normalize(self, imgs: np.ndarray):
        """
        Exolabs data is already in range [0, 1] after channel conversion
        """
        ret = imgs.astype(np.float32)
        return ret

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        if imgs.ndim == 2:
            imgs = imgs[None, None:]
        elif imgs.ndim == 3:
            imgs = imgs[None, :]
        return self._normalize(imgs)


def _test():
    data = S2ExolabData()
    data._load_and_save_dataset(512, 32, convert_channels=True)


if __name__ == "__main__":
    _test()
from datasets.s2data import S2Data
from pathlib import Path
import rasterio
import numpy as np

class S2RawData(S2Data):
    band_to_index = {
        "CLD_PRB": 0,
        "SNW_PRB": 1,
        "B1": 2,
        "B2": 3,
        "B3": 4,
        "B4": 5,
        "B5": 6,
        "B6": 7,
        "B7": 8,
        "B8": 9,
        "B9": 10,
        "B10": 11,
        "B11": 12,
        "B12": 13,
        "B8A": 14
    }
    def __init__(self, dataset_dir = Path("data/raw/"), data_dir = Path("data/raw/raw_processed/"), resolution: int = 10):
        super().__init__(dataset_dir, resolution)
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
         try B4-8-12
         B2 - 3- 12
        """
        bands = list(self.images[image].glob("*.jp2"))
        if len(bands) != 15 and len(l) != 13:
            raise RuntimeError(f"Number of bands is not 13 (raw image) or 15 (raw image + SNW/CLD_PROB): found {len(l)}.")
        bands.sort()
        l = []
        for b in bands:
            with rasterio.open(b, 'r') as f:
                img = f.read(1)
                # check if interpolation is necessary
                img = self._convert_full_img_to_resolution(img, pixel_resolution)
                l.append(img)
        if len(l) == 13:
            l = [np.full_like(l[0], np.nan, dtype=np.double)] * 2 + l
        res = np.stack(l)
        
        return res

    def _standardize(self, imgs: np.ndarray) -> np.ndarray:
        """
        Standardize the images to have 0 mean and 1 std.
        Computes the mean and std for every channel, then standardizes.
        """
        mean = imgs.mean(axis=(0, 2, 3))
        std = imgs.std(axis=(0, 2, 3))
        imgs = (imgs.swapaxes(1, -1) - mean) / std
        return imgs.swapaxes(-1, 1).clip(-5, 5)

    def _normalize(self, imgs: np.ndarray) -> np.ndarray:
        """
        The raw data contains cloud and snow probabilities in the (approximate) range [0, 100] -> normalize,
        the other channels are in (approximate) range [0, 15000] 
        """
        cld_snw_probs = imgs[:, 0:2]
        other_bands = imgs[:, 2:]
        cld_snw_probs = (cld_snw_probs.astype(np.float32) / 100).clip(0, 1)

        other_bands = (other_bands.astype(np.float32) / 15000).clip(0, 1)
        # other_bands = self._standardize(other_bands)
        res = np.hstack([cld_snw_probs, other_bands])
        return res

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        return self._normalize(imgs)


def _test():
    s2data = S2RawData()
    # s2data._load_and_save_dataset(512, 32)

if __name__ == "__main__":
    _test()
from datasets.s2data import S2Data
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))

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

    def __init__(self, dataset_dir=Path("data/raw/"), data_dir=Path("data/raw/raw_processed/"), resolution: int = 10, log_normalize=False):
        super().__init__(dataset_dir, resolution)
        self.log_normalize = log_normalize
        image_paths = list(data_dir.glob("*"))
        for image_path in image_paths:
            date = image_path.name[:8]
            self.images[date] = image_path

    def _load_full_image(self, image_dir: Path | str, pixel_resolution: int = 10) -> np.ndarray:
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
        image_dir = Path(image_dir)
        bands = list(image_dir.glob("*.jp2"))
        if len(bands) != 15 and len(bands) != 13:
            raise RuntimeError(f"Number of bands is not 13 (raw image) or 15 (raw image + SNW/CLD_PROB): found {len(bands)}.")
        bands.sort()
        l = []
        out_size = (self.NATIVE_RESOLUTION * self.NATIVE_DIM) // pixel_resolution
        for b in bands:
            with rasterio.open(b, 'r') as f:
                img = f.read(out_shape=(f.count, out_size, out_size), 
                             resampling=Resampling.bilinear)
                l.append(img.squeeze())
        if len(l) == 13:
            l = [np.full_like(l[0], np.nan, dtype=np.double)] * 2 + l
        res = np.stack(l)
        return res

    def _load_full_image_tif(self, image_path: Path | str, pixel_resolution: int = 10) -> np.ndarray:
        """
        Load a full image as downloaded from the Earth Engine, some of the images had also the QA60 band, so account for 13 or 14 bands. 
        It was downloaded as the last one, so no difference in selected band indeces is needed.
        """
        assert Path(image_path).is_file()
        out_size = (self.NATIVE_RESOLUTION * self.NATIVE_DIM) // pixel_resolution
        with rasterio.open(image_path, 'r') as f:
            img = f.read(out_shape=(f.count, out_size, out_size), resampling=Resampling.bilinear, 
                         window=rasterio.windows.Window(1, 1, self.NATIVE_DIM, self.NATIVE_DIM))
            if len(img) == 13 or len(img) == 14:
                band_indeces = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 8]
            else:
                raise RuntimeError(f"Found strange number of bands: {len(img)}.")
            img = img[band_indeces, :, :]
        l = [np.full_like(img[0:2], np.nan, dtype=np.double), img]
        res = np.concatenate(l)
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
        imgs = self.to_4_dim(imgs)
        cld_snw_probs = imgs[:, 0:2]
        other_bands = imgs[:, 2:]
        if not np.any(cld_snw_probs == np.nan):
            cld_snw_probs = (cld_snw_probs.astype(np.float32) / 100).clip(0, 1)

        other_bands = (other_bands.astype(np.float32) / 15000).clip(0, 1)
        # other_bands = self._standardize(other_bands)
        res = np.hstack([cld_snw_probs, other_bands])
        return res

    def _log_normalize(self, imgs: np.ndarray) -> np.ndarray:
        """
        The raw data contains cloud and snow probabilities in the (approximate) range [0, 100] -> normalize,
        the other channels are in (approximate) range [0, 15000] 
        """
        imgs = self.to_4_dim(imgs)
        cld_snw_probs = imgs[:, 0:2]
        other_bands = imgs[:, 2:]
        if not np.any(cld_snw_probs == np.nan):
            cld_snw_probs = (cld_snw_probs.astype(np.float32) / 100).clip(0, 1)

        other_bands = other_bands.astype(np.float32)

        norm_values = np.load("log_normalization_values.npz")
        a, b, c, d = norm_values["a"], norm_values["b"], norm_values["c"], norm_values["d"]

        other_bands = other_bands.transpose(2, 3, 0, 1)
        log_img = np.log(other_bands)
        log_c = np.log(c)
        log_d = np.log(d)

        other_bands = sigmoid(((log_img - log_c) * (b - a) / (log_d - log_c)) + a)
        other_bands = other_bands.transpose(2, 3, 0, 1).astype(np.float32)

        res = np.hstack([cld_snw_probs, other_bands])
        return res

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        if self.log_normalize:
            return self._log_normalize(imgs)
        else:
            return self._normalize(imgs)

    @classmethod
    def filter_bands(cls, img: np.ndarray, bands: list[str]):
        """
        Filters bands of a single image (3-dim). Output is 3-dimentional [num_bands, height, width].
        """
        img = cls.to_3_dim(img)
        ret = []
        for b in bands:
            if b not in cls.band_to_index:
                raise ValueError(f"Unrecognized band: possible options are {list(cls.band_to_index.keys())}, but got {b}.")
            ret.append(img[cls.band_to_index[b]])
        return np.stack(ret)

def _test():
    s2data = S2RawData()
    # s2data._load_and_save_dataset(512, 32)


if __name__ == "__main__":
    _test()
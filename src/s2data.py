from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

class S2Data(ABC):
    NATIVE_DIM = 10980
    NATIVE_RESOLUTION = 10
    def __init__(self, dataset_dir: str | Path):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.array_locations = {
            10 : self.dataset_dir / "arrays10m/",
            20 : self.dataset_dir / "arrays20m/",
            60 : self.dataset_dir / "arrays60m/"
        }
        self.images : dict[str, Path] = {}

    @abstractmethod
    def _load_full_image(self, image: str, pixel_resolution: int = 10) -> np.ndarray:
        """
        Loads the full image (all bands) in the specified resolution, resizing the bands at different resolution.
        """
        pass

    @abstractmethod
    def _normalize(self, imgs: np.ndarray):
        """
        Normalize the images to have pixels in range [0, 1] of type float
        """
        pass

    def _convert_to_resolution(self, img: np.ndarray, pixel_resolution: int = 10) -> np.ndarray:
        img_dim = int((self.NATIVE_RESOLUTION * self.NATIVE_DIM) / pixel_resolution) 
        if img.shape != (img_dim, img_dim):
            # images must be in shape [height, width, channels]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            elif img.ndim == 3:
                img = img.transpose(1,2,0)
            img = cv2.resize(img[:None], (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
            if img.ndim == 3:
                img = img.transpose(2,0,1).squeeze()
        return img

    def _load_and_save_dataset(self, folder: Path):
        print(f"Loading and saving dataset in {str(folder)}")
        folder.mkdir(exist_ok=True, parents=True)
        for id in tqdm(self.images):
            img = self._load_full_image(id)
            self._save_image(img, folder / id)

    def load_arrays(self, resolution: int = 10, ids_to_load: list[str] = None) -> np.ndarray:
        """
        Loads the full data into memory and returns it
        """
        print(f"Loading arrays with {resolution}m resolution...")
        data = []
        if ids_to_load is None:
            for f in tqdm(list(self.array_locations[resolution].glob("*"))):
                img = np.load(f)
                data.append(img)
        else:
            for id in tqdm(ids_to_load):
                data.append(self.load_array(id, resolution=resolution))
        ret = np.stack(data)
        return ret
    
    def load_array(self, id: str, resolution: int = 10):
        """
        Loads the specified data point into memory and returns it.
        """
        img = np.load(self.array_locations[resolution] / f"{id}.npy")
        return img

    def _create_smaller_resolutions(self, resolutions:list[int] = [20, 60]):
        """
        Loads the images 1 by 1 from the 10m resolution arrays, converts them into the 
        specified resolutions and saves the resulting images.
        """
        print("Converting and saving arrays to smaller resolutions...")
        for res in resolutions:
            self.array_locations[res].mkdir(exist_ok=True)
        original_arrays = list(self.array_locations[10].glob("*"))
        for f in tqdm(original_arrays):
            img = np.load(f)
            for res in resolutions:
                scaled_img = self._convert_to_resolution(img, res)
                self._save_image(scaled_img, self.array_locations[res] / f.name)

    def _save_image(self, image: np.ndarray, path: Path):
        np.save(path, image)

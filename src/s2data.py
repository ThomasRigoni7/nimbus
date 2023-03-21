from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from itertools import product

class S2Data(ABC):
    """
    Base class for all the image data classes, it provides common methods to manipulate the images.
    """
    NATIVE_DIM = 10980
    NATIVE_RESOLUTION = 10
    def __init__(self, dataset_dir: str | Path, resolution: int):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.array_locations = {
            10 : self.dataset_dir / "arrays10m/",
            20 : self.dataset_dir / "arrays20m/",
            60 : self.dataset_dir / "arrays60m/"
        }
        self.images : dict[str, Path] = {}
        self.cut_images : dict[str, Path] = {}
        if self.array_locations[resolution].exists():
            cut_images_paths : list[Path] = list(self.array_locations[resolution].glob("*/*"))
            for path in cut_images_paths:
                if path.is_file():
                    id = str(path.relative_to(path.parent.parent).with_suffix(""))
                    self.cut_images[id] = path

    @abstractmethod
    def _load_full_image(self, image: str, pixel_resolution: int = 10) -> np.ndarray:
        """
        Loads the full image (all bands) in the specified resolution, resizing the bands at different resolution.
        """
        pass

    @abstractmethod
    def _normalize(self, imgs: np.ndarray) -> np.ndarray:
        """
        Normalize the images to have pixels in range [0, 1] of type float
        """
        pass

    @abstractmethod
    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        """
        Runs preprocessing steps on the data after loading np arrays (like normalization/band convertion)

        Returns data in the shape [num_images, num_channels, height, width]
        """
        pass

    def _convert_full_img_to_resolution(self, img: np.ndarray, pixel_resolution: int = 10) -> np.ndarray:
        """
        Converts a full tile image into a specified resolution with interpolation. Will produce wrong results if used with cut images.

        The shape of img must be either [height, width] or [num_channels, height, width].
        Always returns an array with shape [num_channels, height, width].
        """
        if img.ndim > 3:
            raise ValueError(f"img to convert must have 2 or 3 dims, found {img.ndim}.")
        img_dim = int((self.NATIVE_RESOLUTION * self.NATIVE_DIM) / pixel_resolution) 
        # images must be in shape [height, width, channels] for cv2 resize
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        elif img.ndim == 3:
            img = img.transpose(1,2,0)
        if img.shape[0] != img_dim or img.shape[1] != img_dim:
            img = cv2.resize(img, (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3:
            img = img.transpose(2,0,1)
        return img
    
    def _cut_image(self, image: np.ndarray, cut_dim: int = 512, cut_overlap: int = 32) -> dict[str, np.ndarray]:
        """
        Cuts the image in images of size cut_dim and overlap cut_overlap.
        image is a 3-dim array of shape [channels, height, width]
        """
        images = {}
        p = 0
        new_image_coordinates = []
        while p < image.shape[1]:
            p = min(image.shape[1] - cut_dim, p)
            new_image_coordinates.append(p)
            if p == image.shape[1] - cut_dim:
                break
            p += cut_dim - cut_overlap

        for x, y in product(new_image_coordinates, new_image_coordinates):
            cut = image[:, x:x+cut_dim, y:y+cut_dim]
            if cut.sum() != 0:
                images[f"{x}_{y}"] = cut
        return images


    def _load_and_save_dataset(self, cut_dim: int = None, cut_overlap: int = None, resolutions=[10, 20, 60]):
        """
        Loads the dataset from the original images and saves them as arrays in the specified resolutions.
        If cut_dim is specified then cuts the original images and saves the sub-images in a folder.
        """
        print(f"Loading and saving dataset in {[self.array_locations[res] for res in resolutions]}")
        for res in resolutions:
            self.array_locations[res].mkdir(exist_ok=True, parents=True)
        for id in tqdm(self.images):
            full_img = self._load_full_image(id, pixel_resolution=10)
            full_img = self._convert_channels(full_img[None, :]).squeeze(0)
            for res in resolutions:
                img = self._convert_full_img_to_resolution(full_img, res)
                if cut_dim is None:
                    self._save_image(img, self.array_locations[res] / id)
                else:
                    (self.array_locations[res] / id).mkdir(exist_ok=True, parents=True)
                    imgs = self._cut_image(img, cut_dim, cut_overlap)
                    print("Saving in:", self.array_locations[res] / id)
                    for coord, img in imgs.items():
                        self._save_image(img, self.array_locations[res] / id / coord)

    def load_arrays(self, resolution: int = 10, ids_to_load: list[str] = None) -> np.ndarray:
        """
        Loads the specified arrays into memory and returns it. If not specified, load the full data.

        The shape of the returned arrays is [num_images, num_channels, height, width]
        """
        print(f"Loading arrays with {resolution}m resolution...")
        data = []
        if ids_to_load is None:
            for f in tqdm(list(self.array_locations[resolution].glob("*/*"))):
                if f.is_file():
                    img = np.load(f)
                    data.append(img)
        else:
            for id in tqdm(ids_to_load):
                data.append(self.load_array(id, resolution=resolution))
        ret = np.concatenate(data, axis=0)
        return ret
    
    def load_array(self, id: str, resolution: int = 10):
        """
        Loads the specified data point into memory and returns it.

        The shape of the returned array is [1, num_channels, height, width]
        """
        img = np.load(self.array_locations[resolution] / f"{id}.npy")
        if img.ndim == 2:
            img = img[None, None, :]
        elif img.ndim == 3:
            img = img[None, :]
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
                scaled_img = self._convert_full_img_to_resolution(img, res)
                self._save_image(scaled_img, self.array_locations[res] / f.name)

    def _save_image(self, image: np.ndarray, path: Path):
        np.save(path, image)

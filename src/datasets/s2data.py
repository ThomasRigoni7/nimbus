from pathlib import Path
import numpy as np
import torch
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
    def _load_full_image(self, image_path_or_dir: str, pixel_resolution: int = 10) -> np.ndarray:
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

    @classmethod
    def _get_cut_coordinates(cls, image_dim: int, cut_dim: int = 512, cut_overlap: int = 32) -> list[int]:
        """
        Returns the coordinates of the top-left corner of square images to cut with a certain dim and overlap.
        """
        p = 0
        new_image_coordinates = []
        while p < image_dim:
            p = min(image_dim - cut_dim, p)
            new_image_coordinates.append(p)
            if p == image_dim - cut_dim:
                break
            p += cut_dim - cut_overlap
        return new_image_coordinates

    @classmethod
    def _cut_image(cls, image: np.ndarray, cut_dim: int = 512, cut_overlap: int = 32) -> dict[str, np.ndarray]:
        """
        Cuts the image in images of size cut_dim and overlap cut_overlap.
        image is a 3-dim array of shape [channels, height, width]
        """
        images = {}
        image = cls.to_3_dim(image)
        assert image.shape[1] == image.shape[2], "Error: cutting rectangular image, not a square."

        new_image_coordinates = cls._get_cut_coordinates(image.shape[1], cut_dim, cut_overlap)

        for x, y in product(new_image_coordinates, new_image_coordinates):
            cut = image[:, x:x+cut_dim, y:y+cut_dim]
            if cut.sum() != 0:
                images[f"{x}_{y}"] = cut
        return images
    
    @classmethod
    def to_3_dim(cls, img: np.ndarray|torch.Tensor):
        """
        Converts the input array to shape [num_channels, height, width]
        """
        if img.ndim == 3:
            return img
        elif img.ndim == 2:
            return img[None, :]
        elif img.ndim == 4:
            return img.squeeze(axis=0)
        else:
            raise ValueError(f"Cannot convert array to 3 dim: shape is {img.shape}.")
        
    @classmethod
    def to_4_dim(cls, img: np.ndarray|torch.Tensor):
        """
        Converts the input array to shape [num_images, num_channels, height, width]
        """
        if img.ndim == 4:
            return img
        if img.ndim == 2:
            return img[None, None, :]
        elif img.ndim == 3:
            return img[None, :]
        else:
            raise ValueError(f"Cannot convert array to 3 dim: shape is {img.shape}.")

    def _load_dataset_from_images(self, cut_dim: int = None, cut_overlap: int = None, resolutions=[10, 20, 60], convert_channels:bool=False,
                      save_arrays: bool = False, return_arrays: bool = False, ids_to_load:list[int] = None):
        """
        Loads the dataset from the original images, converts them to the specified resolutions.
        If cut_dim is specified then cuts the original images and saves the sub-images in a folder.

        If specified saves the images as arrays or returns the images as a dict where ret[resolution][image_id] = image.
        """
        print(f"Loading dataset...")
        if save_arrays:
            print(f"Saving in {[str(self.array_locations[res]) for res in resolutions]}")
        if return_arrays:
            ret = {}
            for res in resolutions:
                ret[res] = {}
        else:
            ret = None
        for res in resolutions:
            self.array_locations[res].mkdir(exist_ok=True, parents=True)
        iterator = self.images
        if ids_to_load is not None:
            iterator = ids_to_load
        for id in tqdm(iterator):
            for res in resolutions:
                img = self._load_full_image(self.images[id], pixel_resolution=res)
                if convert_channels:
                    img = self._convert_channels(img[None, :]).squeeze(0)
                if cut_dim is None:
                    if save_arrays:
                        self._save_image(img, self.array_locations[res] / id)
                    if return_arrays:
                        ret[res][id] = img
                else:
                    imgs = self._cut_image(img, cut_dim, cut_overlap)
                    if save_arrays:
                        (self.array_locations[res] / id).mkdir(exist_ok=True, parents=True)
                        print("Saving in:", self.array_locations[res] / id)
                        for coord, img in imgs.items():
                            self._save_image(img, self.array_locations[res] / id / coord)
                    if return_arrays:
                        for coord, img in imgs.items():
                            cut_id = str(Path(id) / coord)
                            ret[res][cut_id] = img

        return ret

    def load_arrays(self, resolution: int = 10, ids_to_load: list[str] = None, verbose:bool=True, return_dict:bool=False) -> np.ndarray | dict[str, np.ndarray]:
        """
        Loads the specified arrays into memory and returns it. If not specified, load the full data.

        The returned value is an array of shape [num_images, num_channels, height, width] if return_dict is False (default), 
        otherwise  a dict[str, np.ndarray] where dict[image_id] is an array of shape [num_channels, height, width]
        """
        if verbose:
            print(f"Loading arrays with {resolution}m resolution...")
        data = {}
        if ids_to_load is None:
            iterator = list(self.array_locations[resolution].glob("*"))
            if verbose:
                iterator = tqdm(iterator)
            for f in iterator:
                if f.is_file():
                    img = self.to_3_dim(np.load(f))
                    data[f.with_suffix("").name] = img
        else:
            if verbose:
                ids_to_load = tqdm(ids_to_load)
            for id in ids_to_load:
                data[id] = self.to_3_dim(self.load_array(id, resolution=resolution))
        if not return_dict:
            return np.stack(list(data.values()), axis=0)
        return data
    
    def load_array(self, id: str, resolution: int = 10) -> np.ndarray:
        """
        Loads the specified data point into memory and returns it.

        The shape of the returned array is [1, num_channels, height, width]
        """
        img = np.load(self.array_locations[resolution] / f"{id}.npy")
        return self.to_4_dim(img)
    
    def load_preprocessed_arrays(self, resolution: int = 10, ids_to_load: list[str] = None, verbose:bool=True, return_dict:bool=False):
        """
        Loads the specified arrays into memory, applies preprocess and returns it.

        The returned value is an array of shape [num_images, num_channels, height, width] if return_dict is False (default), 
        otherwise  a dict[str, np.ndarray] where dict[image_id] is an array of shape [num_channels, height, width]
        """
        images = self.load_arrays(resolution = resolution, ids_to_load=ids_to_load, verbose=verbose, return_dict=return_dict)
        if verbose:
            print("Preprocessing images:")
        if return_dict:
            iterator = images.items()
            if verbose:
                iterator = tqdm(iterator)
            for id, img in iterator:
                images[id] = self.to_3_dim(self.preprocess(img))
            return images
        return self.preprocess(images)

    def _save_image(self, image: np.ndarray, path: Path):
        np.save(path, image)

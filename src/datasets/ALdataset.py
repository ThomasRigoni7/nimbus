import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import S2Data, S2RawData
from data_augmentation import SegmentationTransforms
from additional_data import load_additional_data
from itertools import product

class ActiveLearningDataset(Dataset):
    """
    Dataset used to train a model with active learning.

    Parameters:
        - images: dict[str, ndarray] -> dictionary of preprocessed full images of shape [channels, height, width]
        - labels: dict[str, ndarray] -> dictionary of full-size segmentation indexes of shape [height, width]
        - image_dim: int -> size of the cut images returned by the dataset.
        - training_ids: list[str] -> list of the indeces used to start training the dataset.
        - test_labels: dict[str, ndarray] -> dictionary of test set labels. The image set must be disjoined with training_ids.
        - transforms: SegmentationTransforms -> transformations for training data augmentation.
        - additional_layers: list[str] -> list of str keys of the additional data dict: choose from:
            ["lakes", "glaciers", "surface_water", "altitude", "landcover", "treecover"]
    """

    def __init__(self,
                 images: dict[str, np.ndarray],
                 labels: dict[str, np.ndarray],
                 image_dim: int,
                 training_ids: list[str],
                 test_labels: dict[str, np.ndarray],
                 input_bands: list[str] = ["B12", "B8", "B4"],
                 transforms: SegmentationTransforms = None, 
                 additional_layers: list[str] = []):
        super().__init__()
        self.images = images
        self.labels = labels
        self.img_dim = image_dim
        if type(input_bands) == str and input_bands == "all":
            input_bands = ["B1", "B2", "B3", "B4","B5","B6","B7","B8", "B8A", "B9","B10","B11","B12"]
        self.input_bands = input_bands
        self.training_ids = training_ids
        self.test_labels = test_labels
        self.transforms = transforms
        self.additional_data = load_additional_data()
        self.additional_layers = additional_layers
        self._check_input_consistency()
        self.update_indexes()

    def _check_input_consistency(self):
        """
        Check that all input images and labels have the same size, the bands and the additional layers are correct.
        """
        print("Checking input consistency...")
        self.original_img_dim = self.images[self.training_ids[0]].shape[-1]
        for id, img in self.images.items():
            assert img.shape[-1] == self.original_img_dim and img.shape[-2] == self.original_img_dim, f"Encountered image with different size than the other ones: expected {self.original_img_dim}, found shape {img.shape}."
            assert self.labels[id].shape[-1] == self.original_img_dim and self.labels[id].shape[-2] == self.original_img_dim, f"Encountered label with different size than the other ones: expected {self.original_img_dim}, found shape {img.shape}."
        assert all(b in list(S2RawData.band_to_index.keys()) for b in self.input_bands), f"Specified bands do not exist: possible choices: {list(S2RawData.band_to_index.keys())}"
        assert all([l in self.additional_data.keys() for l in self.additional_layers]), f"Specified layers not in additional data: possible choices: {list(self.additional_data.keys())}"
        # TODO: check test/train image set is disjoined.
        print("Input consistency checks passed!")
    
    def _get_position(self, index: int) -> tuple[int, int]:
        id: str = self.index2id[index]
        h_w = id.split("/")[1]
        h, w = h_w.split("_")
        return int(h), int(w)

    def _apply_cut(self, img, h, w, dim):
        """
        Cuts the given image in a square starting from the top-left corner at h-w, of size dim x dim.

        Returns a 3-dim image of shape [num_channels, height, width]
        """
        img = S2Data.to_3_dim(img)
        return img[:, h:h+dim, w:w+dim]

    def update_indexes(self):
        """
        Calculates the indexes based on the training_ids and the input dim.
        
        The final indexes are of the form {image_id}/{height}_{width}
        """
        cut_coordinates = S2Data._get_cut_coordinates(self.original_img_dim, cut_dim=self.img_dim)
        self.index2id = []
        for id in self.training_ids:
            for h, w in product(cut_coordinates, cut_coordinates):
                cut_id = f"{id}/{h}_{w}"
                self.index2id.append(cut_id)

    def update_training_ids(self, new_ids: list[str]):
        """
        Updates the training ids by adding the new_ids passed as argument.
        """
        self.training_ids += new_ids
        self.update_indexes()

    def update_labels(self, new_labels: dict[str, np.ndarray]):
        """
        Updates the labels with the new masks provided.
        """
        self.labels.update(new_labels)

    def compute_loss_mask(self, img: np.ndarray):
        """
        Given a single image (only raw bands), computes the loss mask to apply. It is computed by checking if any 
        reflectance value is exactly equal to 0.
        """
        if img.ndim != 3:
            raise ValueError(f"Image ndim must be 3 [num_channels, height, width] for computing loss mask, found shape: {img.shape}")
        if len(img) == 15:
            img = img[2:]
        elif len(img) != 13:
            raise ValueError("Cannot compute the loss mask of a raw image with a number of channels different from 15 or 13: cannot remove snow/cloud probability masks.")

        mask = np.any(img == 0, axis=0)            
        return mask

    def __len__(self):
        return len(self.index2id)

    def __getitem__(self, index) -> tuple[str, torch.Tensor, torch.Tensor]:
        """
        Returns the item corresponding to index as a tuple of (id, image, label)
        """
        # select right image and label
        cut_id = self.index2id[index]
        id = cut_id.split("/")[0]
        image = self.images[id]
        label = self.labels[id]
        
        # cut label and additional data
        h, w = self._get_position(index)
        cut_image = self._apply_cut(image, h, w, self.img_dim)
        cut_label = self._apply_cut(label, h, w, self.img_dim)

        additional_layers = []
        for layer_name in self.additional_layers:
            layer = self.additional_data[layer_name]
            cut_layer = self._apply_cut(layer, h, w, self.img_dim)
            additional_layers.append(S2Data.to_3_dim(cut_layer))
        
        # filter image bands and add additional layers
        loss_mask = self.compute_loss_mask(cut_image)
        cut_image = S2RawData.filter_bands(cut_image, self.input_bands)
        cut_image = np.concatenate([cut_image] + additional_layers, axis=0)

        # img, label = self._apply_transforms(cut_image, cut_label)
        return id, cut_image, cut_label, loss_mask



class AugmentedDataset(Dataset):
    """
    Simple dataset that returns directly the queried element and only applies data augmentation if required.
    """
    pass

class dataLoaderDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.raw_data = S2RawData()
        self.image_paths = self.raw_data.images
        self.index2id = list(self.image_paths.keys())

    def __len__(self):
        return len(self.index2id)
    
    def __getitem__(self, index) -> np.ndarray:
        # print("loading", index)
        id = self.index2id[index]
        array = self.raw_data.load_preprocessed_arrays(ids_to_load=[id], return_dict=True, verbose=False, resolution=10)
        # print("loaded", index)
        return array

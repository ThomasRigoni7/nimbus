from datasets.s2data import S2Data
from datasets.s2rawdata import S2RawData
from data_augmentation import SegmentationTransforms
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class S2Dataset(Dataset):
    """
    Base class for the dataset classes, the __getitem__ method returns a list with the sample from all datasets provided.
    """

    def __init__(self,
                 raw_data: S2RawData,
                 other_dataset_list: list[S2Data],
                 resolution: int = 10,
                 load_into_memory: bool = False,
                 use_cut_images: bool = True,
                 transforms: SegmentationTransforms = None):
        super().__init__()
        self.images = {}
        self.cut_images = {}
        self.use_cut_images = use_cut_images
        self.in_memory = load_into_memory
        self.resolution = resolution
        self.transforms = transforms
        self.dataset_list: list[S2Data] = [raw_data] + other_dataset_list
        for id in raw_data.images:
            images = {}
            for dataset in self.dataset_list:
                images[dataset.__class__.__name__] = dataset.images.get(id, None)
            if None not in images.values():
                self.images[id] = images
        for id in raw_data.cut_images:
            cut_images = {}
            for dataset in self.dataset_list:
                image_path = dataset.cut_images.get(id, None)
                cut_images[dataset.__class__.__name__] = image_path
                if image_path is None:
                    print("None with ID:", id)
                    print(list(cut_images.values()))
            if None in list(cut_images.values()):
                print("Not added ID:", id)
            else:
                self.cut_images[id] = cut_images
        self.image_ids = list(self.images.keys())
        self.image_ids.sort()
        print(len(self.image_ids))
        self.cut_image_ids = list(self.cut_images.keys())
        self.cut_image_ids.sort()

        # load the full data with the specified resolution
        if load_into_memory:
            self.data = {}
            if use_cut_images:
                for dataset in self.dataset_list:
                    self.data[dataset.__class__.__name__] = dataset.load_preprocessed_arrays(resolution=resolution,
                                                                                             ids_to_load=self.cut_image_ids)
            else:
                for dataset in self.dataset_list:
                    self.data[dataset.__class__.__name__] = dataset.load_preprocessed_arrays(resolution=resolution,
                                                                                             ids_to_load=self.image_ids)

    def __len__(self):
        if self.use_cut_images:
            return len(self.cut_images)
        else:
            return len(self.images)

    def _apply_transforms(self, data: torch.Tensor, labels: torch.Tensor):
        if self.transforms is None:
            return data, labels
        return self.transforms.apply(data, labels)

    def eval(self):
        self.transforms.set_train(False)

    def train(self):
        self.transforms.set_train(True)

    def _get_label_channel_pixel_counts(self, labels: np.ndarray) -> np.ndarray:
        """
        Given labels of a single or multiple images returns the count of pixels belonging to each class (max value)

        Accepts an input ndarray of labels of shape [n_samples, n_channels, heights, width] and returns
        an ndarray of shape [n_channels]
        """
        assert labels.ndim == 4
        n_channels = labels.shape[1]
        max_label = np.argmax(labels, axis=1)
        counts = []
        for c in range(n_channels):
            count = np.count_nonzero(max_label == c)
            counts.append(count)
        return np.array(counts, dtype=np.int64)

    def __getitem__(self, index) -> list[np.ndarray]:
        """
        Returns the full data (all datasets) for the sample in a list of 3-dim torch tensors: [channels, height, width]
        """
        ret = []
        if self.in_memory:
            for dataset in self.dataset_list:
                ret.append(self.data[dataset.__class__.__name__][index])
        else:
            # print("Image:", self.cut_image_ids[index])
            for dataset in self.dataset_list:
                if self.use_cut_images:
                    data = dataset.load_array(resolution=self.resolution, id=self.cut_image_ids[index])
                    ret.append(dataset.preprocess(data).squeeze(axis=0))
                else:
                    data = dataset.load_array(resolution=self.resolution, id=self.image_ids[index])
                    ret.append(dataset.preprocess(data).squeeze(axis=0))

        return ret


def _test():
    from s2cloudlessdata import S2CloudlessData
    from s2exolabsdata import S2ExolabData
    raw_data = S2RawData()
    cloudless_data = S2CloudlessData()
    exolabs_data = S2ExolabData()
    dataset = S2Dataset(raw_data, [cloudless_data, exolabs_data])
    img = dataset[40]
    for set in img:
        print(set.shape)


if __name__ == "__main__":
    _test()
import torch
from datasets.s2rawdata import S2RawData
from datasets.s2cloudlessdata import S2CloudlessData
from datasets.s2exolabsdata import S2ExolabData
from datasets.s2dataset import S2Dataset
from data_augmentation import SegmentationTransforms
import numpy as np
from tqdm import tqdm

class S2RawCloudlessExolabDataset(S2Dataset):
    """
    Dataset that returns RGB values as data, 3-dim labels of dim [cloud, no-snow, snow] are combined from S2cloudless and exolabs.
    """
    def __init__(self, bands:list[str]=["B4", "B3", "B2"], resolution:int=10, load_into_memory:bool=False, transforms: SegmentationTransforms = None):
        self.raw_data = S2RawData(resolution=resolution)
        self.cloudless_data = S2CloudlessData(resolution=resolution)
        self.exolabs_data = S2ExolabData(resolution=resolution)
        super().__init__(self.raw_data, [self.cloudless_data, self.exolabs_data], resolution=resolution, load_into_memory=load_into_memory, 
                         transforms=transforms)
        self.bands = bands
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple containing (RGB image, label), where the RGB image is taken from the raw data,
        the label is 3-dimentional [cloud, no-snow, snow] built using s2cloudless and Exolabs snow classification.
        """
        sample_data = super().__getitem__(index)
        raw, cloudless, exolabs = sample_data
        raw_bands = []
        for b in self.bands:
            if b not in self.raw_data.band_to_index:
                raise ValueError(f"Unrecognized band: possible options are {list(self.raw_data.band_to_index.keys())}, but got {b}.")
            raw_bands.append(raw[self.raw_data.band_to_index[b]])
        

        data = torch.from_numpy(np.stack(raw_bands))
        label = torch.from_numpy(np.concatenate([cloudless, exolabs], axis=0))

        data, label = self._apply_transforms(data, label)

        return data, label
    
    def get_class_weights(self):
        """
        Returns the class weights to use in the CrossEntropy loss as a torch Tensor:
        
        The weight of a class is inversely proportional to the number of pixels belonging to the class (in the labels) 
        and the sum of the weights is 1.
        """
        print("Computing class weights...")
        if self.in_memory:
            cloudless_labels = self.data[self.cloudless_data.__class__.__name__]
            exolabs_labels = self.data[self.exolabs_data.__class__.__name__]
            labels = np.concatenate([cloudless_labels, exolabs_labels], axis=1)
            counts = self._get_label_channel_pixel_counts(labels)
        else:
            counts = np.zeros((3))
            if self.use_cut_images:
                ids = self.cut_image_ids
            else:
                ids = self.image_ids
            
            for id in tqdm(ids):
                cloudless_label = self.cloudless_data.load_preprocessed_arrays(resolution=self.resolution, ids_to_load=[id], verbose=False)
                exolabs_label = self.exolabs_data.load_preprocessed_arrays(resolution=self.resolution, ids_to_load=[id], verbose=False)
                labels = np.concatenate([cloudless_label, exolabs_label], axis=1)
                counts += self._get_label_channel_pixel_counts(labels)
        weights = 1 / counts
        weights = weights / weights.sum()
        return torch.from_numpy(weights)
        





def _test():
    import matplotlib.pyplot as plt
    s2data = S2RawCloudlessExolabDataset()
    rgb, label = s2data[40]
    print(rgb.shape)
    print(label.shape)
    plt.imsave("rgb.jpg", rgb.permute(1, 2, 0).numpy())

if __name__ == "__main__":
    _test()


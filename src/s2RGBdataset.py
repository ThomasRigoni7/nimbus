import torch
from s2rawdata import S2RawData
from s2cloudlessdata import S2CloudlessData
from s2exolabsdata import S2ExolabData
from s2dataset import S2Dataset
import numpy as np

class S2RGBDataset(S2Dataset):
    """
    Dataset that returns RGB values as data, 3-dim labels of dim [cloud, no-snow, snow] are combined from S2cloudless and exolabs.
    """
    def __init__(self, resolution:int=10, load_into_memory:bool=False):
        raw_data = S2RawData(resolution=resolution)
        cloudless_data = S2CloudlessData(resolution=resolution)
        exolabs_data = S2ExolabData(resolution=resolution)
        super().__init__(raw_data, [cloudless_data, exolabs_data], resolution=resolution, load_into_memory=load_into_memory)

    def __len__(self):
        if self.use_cut_images:
            return len(self.cut_images)
        else:
            return len(self.images)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple containing (RGB image, label), where the RGB image is taken from the raw data,
        the label is 3-dimentional [cloud, no-snow, snow] built using s2cloudless and Exolabs snow classification.
        """
        sample_data = super().__getitem__(index)
        raw, cloudless, exolabs = sample_data
        rgb = raw[4:1:-1]
        label = np.concatenate([cloudless, exolabs], axis=0)

        return torch.from_numpy(rgb.copy()), torch.from_numpy(label)



def _test():
    import matplotlib.pyplot as plt
    s2data = S2RGBDataset()
    rgb, label = s2data[40]
    print(rgb.shape)
    print(label.shape)
    plt.imsave("rgb.jpg", rgb.permute(1, 2, 0).numpy())

if __name__ == "__main__":
    _test()


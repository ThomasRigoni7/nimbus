import torch
from s2rawdata import S2RawData
from s2cloudlessdata import S2CloudlessData
from s2exolabsdata import S2ExolabData
from s2dataset import S2Dataset
import numpy as np

class S2RGBDataset(S2Dataset):
    """
    Dataset that each iteration returns a tuple containing (RGB image, label), where the RGB image is taken from the raw data,
    the labels 3-dimentional [] are built using s2cloudless and Exolabs snow classification.
    """
    def __init__(self):
        raw_data = S2RawData()
        cloudless_data = S2CloudlessData()
        exolabs_data = S2ExolabData()
        super().__init__(raw_data, cloudless_data, exolabs_data)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> torch.Tensor:
        sample_data = super().__getitem__(index)
        raw, cloudless, exolabs = sample_data
        
        rgb = raw[4:1:-1]
        s = self.exolabs_data._convert_channels(exolabs)
        no_snow, snow = s[0], s[1]
        label = np.stack([cloudless, snow, no_snow])

        return rgb, label



def _test():
    import matplotlib.pyplot as plt
    s2data = S2RGBDataset()
    rgb, label = s2data[1]
    plt.imsave("rgb.jpg", rgb.transpose(1, 2, 0))

if __name__ == "__main__":
    _test()


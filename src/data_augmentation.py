import torch
import numpy as np
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

class SegmentationTransforms:
    def __init__(self, crop: bool, hflip: bool, vflip: bool, crop_size:int = 256):
        self.crop = crop
        self.hflip = hflip
        self.vflip = vflip
        self.crop_size = crop_size
        self.train = True
        self.warning_done = False
        
    def set_train(self, train: bool):
        self.train = train
    
    def apply(self, data: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray):
        if (not data.shape == labels.shape) and not self.warning_done:
            response = input(f"""WARNING: Transforming data and labels with different shape!\nData shape:   {data.shape}\nLabels shape: {labels.shape}.\nContinue y/n?(n)""")
            self.warning_done = True
            if response.lower() != "y":
                exit()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if self.train:
            if self.crop:
                i, j, h, w = RandomCrop.get_params(data, (self.crop_size, self.crop_size))
                data = TF.crop(data, i, j, h, w)
                labels = TF.crop(labels, i, j, h, w)
            if self.hflip:
                if torch.rand(1) < 0.5:
                    data = TF.hflip(data)
                    labels = TF.hflip(labels)
            if self.vflip:
                if torch.rand(1) < 0.5:
                    data = TF.vflip(data)
                    labels = TF.vflip(labels)
        else:
            if self.crop:
                data = TF.center_crop(data, self.crop_size)
                labels = TF.center_crop(labels, self.crop_size)
        
        return data, labels

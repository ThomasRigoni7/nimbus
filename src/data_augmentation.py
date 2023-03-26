import torch
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

class SegmentationTransforms:
    def __init__(self, crop: bool, hflip: bool, vflip: bool, crop_size:int = 256):
        self.crop = crop
        self.hflip = hflip
        self.vflip = vflip
        self.crop_size = crop_size
        self.train = True
        
    def set_train(self, train: bool):
        self.train = train
    
    def apply(self, data: torch.Tensor, labels: torch.Tensor):
        assert data.shape == labels.shape, "ERROR: Transforming data and labels with different shape!"
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

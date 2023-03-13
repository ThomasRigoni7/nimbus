import torch
from s2data import S2Data
from s2rawdata import S2RawData
from s2cloudlessdata import S2CloudlessData
from s2exolabsdata import S2ExolabData
from torch.utils.data import Dataset

class S2Dataset(Dataset):
    def __init__(self, raw_data: S2RawData, cloudless_data: S2CloudlessData, exolabs_data: S2ExolabData, resolution: int=10, load_into_memory: bool=False):
        super().__init__()
        self.images = {}
        self.in_memory = load_into_memory
        self.resolution = resolution
        self.raw_data = raw_data
        self.cloudless_data = cloudless_data
        self.exolabs_data = exolabs_data
        self.dataset_list: list[S2Data] = [raw_data, cloudless_data, exolabs_data]
        for id in raw_data.images:
            self.images[id] = {}
            for dataset in self.dataset_list:
                self.images[id][dataset.__class__.__name__] = dataset.images[id]
        self.image_ids = list(self.images.keys())

        # load the full data with 60m resolution so that it fits into memory
        if load_into_memory:
            self.data = {}
            for dataset in self.dataset_list:
                data = dataset.load_arrays(resolution=resolution, ids_to_load=list(self.images.keys()))
                self.data[dataset.__class__.__name__] = dataset.preprocess(data)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> torch.Tensor:
        """
        Returns the full data (all datasets) for the sample
        """
        ret = []
        if self.in_memory:
            for dataset in self.dataset_list:
                ret.append(self.data[dataset.__class__.__name__][index])
        else:
            for dataset in self.dataset_list:
                ret.append(dataset.load_array(resolution=self.resolution, id=self.image_ids[index]))

        return ret

from s2data import S2Data
from s2rawdata import S2RawData
from torch.utils.data import Dataset
import numpy as np

class S2Dataset(Dataset):
    def __init__(self, raw_data: S2RawData, other_dataset_list:list[S2Data], resolution: int=10, load_into_memory: bool=False, use_cut_images:bool = True):
        super().__init__()
        self.images = {}
        self.cut_images = {}
        self.use_cut_images = use_cut_images
        self.in_memory = load_into_memory
        self.resolution = resolution
        self.dataset_list: list[S2Data] = [raw_data] + other_dataset_list
        for id in raw_data.images:
            self.images[id] = {}
            for dataset in self.dataset_list:
                self.images[id][dataset.__class__.__name__] = dataset.images[id]
        for id in raw_data.cut_images:
            self.cut_images[id] = {}
            for dataset in self.dataset_list:
                self.cut_images[id][dataset.__class__.__name__] = dataset.cut_images[id]
        self.image_ids = list(self.images.keys())
        self.cut_image_ids = list(self.cut_images.keys())

        # load the full data with the specified resolution
        if load_into_memory:
            self.data = {}
            if use_cut_images:
                for dataset in self.dataset_list:
                    data = dataset.load_arrays(resolution=resolution, ids_to_load=[str(path) for path in self.cut_images.keys()])
                    self.data[dataset.__class__.__name__] = dataset.preprocess(data)
            else:
                for dataset in self.dataset_list:
                    data = dataset.load_arrays(resolution=resolution, ids_to_load=list(self.images.keys()))
                    self.data[dataset.__class__.__name__] = dataset.preprocess(data)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> list[np.ndarray]:
        """
        Returns the full data (all datasets) for the sample in a list of 3-dim ndarrays: [channels, height, width]
        """
        ret = []
        if self.in_memory:
            for dataset in self.dataset_list:
                ret.append(self.data[dataset.__class__.__name__][index])
        else:
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
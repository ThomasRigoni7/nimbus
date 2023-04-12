from datasets import S2RawData, S2CloudlessData, S2ExolabData

raw_data = S2RawData()
raw_data._load_dataset_from_images(save_arrays=True)

cloudless_data = S2CloudlessData()
cloudless_data._load_dataset_from_images(save_arrays=True)

exolabs_data = S2ExolabData()
exolabs_data._load_dataset_from_images(convert_channels=True, save_arrays=True)
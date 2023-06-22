from datasets import S2RawData
import matplotlib.pyplot as plt

raw_data = S2RawData()
image = raw_data.load_arrays(ids_to_load=["20210312"]).squeeze()
band_names = list(raw_data.band_to_index.keys())[2:]


for b in range(len(image) - 2):
    raw_data.band_to_index
    plt.hist(image[b+2].flatten(), bins=400, label=band_names[b], histtype="step")
    
plt.xlabel("band value")
plt.ylabel("pixel count")
plt.legend()
plt.show()

raw_data = S2RawData(resolution=60, log_normalize=True)
image = raw_data.load_preprocessed_arrays(ids_to_load=["20210312"], resolution=10).squeeze()

for b in range(len(image) - 2):
    plt.hist(image[b+2].flatten(), bins=400, label=band_names[b], histtype="step")
    

plt.xlabel("band value")
plt.ylabel("pixel count")
plt.legend()
plt.show()
import torch
import torch.nn as nn
from segnet import SegNetLite
from s2RGBdataset import S2RGBDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = S2RGBDataset()
dataloader = DataLoader(dataset, 24)
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegNetLite()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), 1e-3)

for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}:")
    for batch in tqdm(dataloader):
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)
        preds = model(data)
        
        l = loss_fn(preds, labels)
        l.backward()
        optimizer.step()

first_image = dataset[0]
data, labels = first_image
preds = model(data)
plt.imsave("data.png", data)
plt.imsave("prediction.png", preds)
plt.imsave("label.png", labels)



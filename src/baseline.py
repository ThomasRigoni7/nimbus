import torch
from segnet import SegNetLite
from s2RGBdataset import S2RGBDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pytorch_lightning as pl

dataset = S2RGBDataset(resolution=60, load_into_memory=False)
train_dataset, val_dataset = random_split(dataset, [0.7, 0.3])
train_dataloader = DataLoader(train_dataset, 24)
val_dataloader = DataLoader(val_dataset, 24)
model = SegNetLite()

trainer = pl.Trainer(logger=None, max_epochs=30, gpus=1)
trainer.fit(model, train_dataloader, val_dataloader)


first_image = dataset[0]
device = torch.device("cuda")
model = model.to(device)
data, labels = first_image
data = data.to(device)
labels = labels.to(device)
preds = model(data[None,:]).squeeze()
plt.imsave("data.png", data.permute(1, 2, 0).cpu().detach().numpy())
plt.imsave("prediction.png", preds.permute(1, 2, 0).clip(0, 1).cpu().detach().numpy())
plt.imsave("label.png", labels.permute(1, 2, 0).cpu().detach().numpy())
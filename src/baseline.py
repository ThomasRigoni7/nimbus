import torch
import torch.nn as nn
from segnet import SegNetLite
from s2RGBdataset import S2RGBDataset
from s2RGBCEdataset import S2RGBCloudlessExolabDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np

class DeepLabV3Baseline(pl.LightningModule):
    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
        self.regressor = nn.Conv2d(21, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.backbone(x)
        return self.regressor(out["out"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        pred = self(data)
        loss = torch.nn.functional.cross_entropy(pred, label)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)

def save_img(path: str, image: torch.Tensor):
    image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if image.shape[-1] == 2:
        shape = image.shape
        zero = np.zeros((shape[0], shape[1], 1))
        image = np.concatenate([image, zero], axis=-1)
    plt.imsave("results/prova/" + path, image)

# dataset = S2RGBDataset(resolution=60, load_into_memory=False)
dataset = S2RGBCloudlessExolabDataset(resolution=60, load_into_memory=False)
train_dataset, val_dataset = random_split(dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_dataset, 8)
val_dataloader = DataLoader(val_dataset, 8, shuffle=False)
# model = SegNetLite(out_classes=3)
model = DeepLabV3Baseline(3)

trainer = pl.Trainer(logger=None, max_epochs=50, accelerator="gpu", devices=1)
trainer.fit(model, train_dataloader, val_dataloader)
# model = model.load_from_checkpoint("checkpoints/epoch=49-step=1400.ckpt")

def save_predictions(dataset, model, num_images, folder):
    sequential_loader = DataLoader(dataset, batch_size=num_images, shuffle=False)
    for batch in sequential_loader:
        first_images = batch
        break
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    data, labels = first_images
    data = data.to(device)
    labels = labels.to(device)
    preds = model(data)
    print(preds.max())
    print(preds.min())

    from pathlib import Path
    Path("results/prova/train").mkdir(exist_ok=True, parents=True)
    Path("results/prova/val").mkdir(exist_ok=True, parents=True)
    for i in range(num_images):
        save_img(f"{folder}/data{i}.png", data[i])
        save_img(f"{folder}/prediction{i}.png", preds[i].clip(0, 1))
        save_img(f"{folder}/label{i}.png", labels[i])

save_predictions(train_dataset, model, 10, "train")
save_predictions(val_dataset, model, 10, "val")
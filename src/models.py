import torch
import torch.nn as nn
import pytorch_lightning as pl

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
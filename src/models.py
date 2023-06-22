import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


class SegNetLite(pl.LightningModule):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], 
            in_channels:int=3, out_classes:int = 3, dropout_prob: float = 0.2):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        # Construct downsampling layers.
        # Blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        down_filter_sizes_extended = [in_channels] + down_filter_sizes
        layers_conv_down = [ nn.Conv2d(down_filter_sizes_extended[i], 
                                down_filter_sizes_extended[i+1], 
                                kernel_sizes[i], 
                                1, 
                                conv_paddings[i]) 
                                for i in range(self.num_down_layers)]
        layers_bn_down = [nn.BatchNorm2d(down_filter_sizes[i]) for i in range(self.num_down_layers)]
        layers_pooling = [nn.MaxPool2d(pooling_kernel_sizes[i], pooling_strides[i], return_indices=True) for i in range(self.num_down_layers)]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # Blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        up_filter_sizes_extended = [down_filter_sizes[-1]] + up_filter_sizes
        layers_conv_up = [nn.Conv2d(up_filter_sizes_extended[i],
                            up_filter_sizes_extended[i+1], 
                            kernel_sizes[self.num_up_layers - i - 1], 
                            1, 
                            conv_paddings[self.num_up_layers - i -1]) for i in range(self.num_down_layers)]
        layers_bn_up = [nn.BatchNorm2d(up_filter_sizes[i]) for i in range(self.num_up_layers)]
        layers_unpooling = [nn.MaxUnpool2d(pooling_kernel_sizes[self.num_up_layers - i - 1],
                                pooling_strides[self.num_up_layers - i - 1]) for i in range(self.num_down_layers)]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.conv1x1 = nn.Conv2d(up_filter_sizes[-1], out_classes, 1)


    def forward(self, x):
        # forward pass the down modules
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        pooling_indices = []
        for i in range(self.num_down_layers):
            x = self.layers_conv_down[i](x)
            x = self.layers_bn_down[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            x, indices = self.layers_pooling[i](x)
            pooling_indices.append(indices)

        # up modules
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        for i in range(self.num_up_layers):
            x = self.layers_unpooling[i](x, pooling_indices[self.num_down_layers - i - 1])
            x = self.layers_conv_up[i](x)
            x = self.layers_bn_up[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # last convolution for classification:
        x = self.conv1x1(x)
        return x
    

class PLModel(pl.LightningModule):
    def __init__(self, model: nn.Module, out_classes: int, class_weights: torch.Tensor, lr:float=1e-3):
        super(PLModel, self).__init__()
        self.model = model
        self.lr = lr
        self.miou = JaccardIndex("multiclass", num_classes=out_classes)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def common_step(self, batch):
        data, label = batch
        print(data.shape)
        pred = self.model(data)
        loss = self.criterion(pred.float(), label)
        loss = loss.mean()
        miou = self.miou(pred, label)
        return loss, miou

    def training_step(self, batch, batch_idx):
        loss, miou = self.common_step(batch)
        self.log("train loss", loss)
        self.log("train miou", miou)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, miou = self.common_step(batch)
        self.log("val loss", loss)
        self.log("val miou", miou)
        return loss
    
    def test_step(self, batch, batch_id):
        loss, miou = self.common_step(batch)
        self.log("test loss", loss)
        self.log("test miou", miou)
        return loss

class ModelWithIndex(PLModel):
    def __init__(self, model: str, in_channels: int, out_classes: int, class_weights: torch.Tensor, model_args: dict = {}, dropout:float=0, lr:float=1e-3):
        self.save_hyperparameters()
        if model == "segnet":
            model_args.update({"dropout_prob": dropout})
            backbone = SegNetLite(in_channels=in_channels, out_classes=out_classes, **model_args)
        elif model == "deeplabv3":
            backbone = smp.DeepLabV3Plus(in_channels=in_channels, classes=out_classes, **model_args, encoder_weights=None)
        elif model == "unet":
            backbone = smp.UnetPlusPlus(in_channels=in_channels, classes=out_classes, **model_args, encoder_weights=None)
        elif model == "fpn":
            model_args.update({"decoder_dropout": dropout})
            backbone = smp.FPN(in_channels=in_channels, classes=out_classes, **model_args, encoder_weights=None)
        elif model == "psp":
            model_args.update({"psp_dropout": dropout})
            backbone = smp.PSPNet(in_channels=in_channels, classes=out_classes, **model_args, encoder_weights=None)

        super().__init__(backbone, out_classes, class_weights, lr)
        self.ensemble_dim = 0
        self.return_predict_probabilities = False

    def set_ensemble_dim(self, dim: int):
        print(f"Setting ensemble dim to {dim}!")
        self.ensemble_dim = dim

    def set_dropout(self, active: bool):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                if active:
                    module.train()
                else:
                    module.eval()
                
    def common_step(self, batch):
        ids, data, label = batch
        pred = self.model(data)
        loss = self.criterion(pred.float(), label.to(dtype=torch.long))
        loss = loss.mean()
        miou = self.miou(pred, label)
        return loss, miou
    
    def ensemble_step(self, batch):
        """
        Calculates the various Monte Carlo (with dropout) inferences, 
        from these calculates the uncertainty mask and the total sample uncertainty.

        Returns a tuple[dict[str, torch.Tensor], dict[str, float]] containing the uncertainty masks and values.
        """
        ids, data, labels = batch
        del labels
        self.set_dropout(True)
        predictions = {id:[] for id in ids}
        for _ in range(self.ensemble_dim):
            preds = self.model(data)
            for i, id in enumerate(ids):
                predictions[id].append(preds[i].cpu().detach())
        del data
        uncertainty_masks = {}
        uncertainty_values = {}
        for id, preds in predictions.items():
            mask = torch.softmax(torch.stack(preds).to(self.device), dim=1).std(dim=0).mean(dim=0)
            sample_uncertainty = 2 * mask.sum() / (mask.shape[0] * mask.shape[1])
            uncertainty_masks[id] = mask.cpu().to(dtype=torch.float16)
            uncertainty_values[id] = sample_uncertainty.cpu().item()

        self.set_dropout(False)
        return uncertainty_masks, uncertainty_values

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """
        If self.ensemble_dim > 0 then runs the ensemble step with MC inference.

        If self.ensemble_dim == 0 then calculates the regular inference on the full model without dropout.
        """
        
        if self.ensemble_dim > 0:
            return self.ensemble_step(batch)
        elif self.ensemble_dim == 0:
            ids, data, labels = batch
            preds = self.model(data)
            segmentation_masks = torch.argmax(preds, dim=1).to(dtype=torch.uint8)
            # show_image(data[0][:3], labels[0], segmentation_masks[0])
            assert len(ids) == len(data) == len(labels) == len(segmentation_masks), f"Error: different lengths. \nids: {len(ids[0])}, data: {len(data)}, labels: {len(labels)}, seg mask: {len(segmentation_masks)}"
            if self.return_predict_probabilities:
                ret = {id:p for id, p in zip(ids, preds.cpu())}
            else:
                ret = {id:p for id, p in zip(ids, segmentation_masks.cpu())}
            return ret
        else:
            raise ValueError(f"ensemble_dim must be >= 0! Found {self.ensemble_dim}")
    
def show_image(img: torch.Tensor, label: torch.Tensor, pred_mask: torch.Tensor):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.set_title("image")
    ax1.imshow(img.cpu().detach().numpy().transpose([1, 2, 0]))
    ax2.set_title("label")
    ax2.imshow(label.cpu().detach().numpy())
    ax3.set_title("pred")
    ax3.imshow(pred_mask.cpu().detach().numpy())
    plt.show()
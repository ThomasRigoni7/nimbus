import torch
from data_augmentation import SegmentationTransforms
from models import SegNetLite
from datasets import S2RawCloudlessExolabDataset, S2Dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np

def save_img(path: str, image: torch.Tensor):
    image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if image.shape[-1] == 2:
        shape = image.shape
        zero = np.zeros((shape[0], shape[1], 1))
        image = np.concatenate([image, zero], axis=-1)
    plt.imsave("results/prova/" + path, image)


transforms = SegmentationTransforms(True, True, True)
# dataset = S2RGBDataset(resolution=60, load_into_memory=False)
# dataset = S2RawCloudlessExolabDataset(bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"], resolution=60, load_into_memory=True)
dataset = S2RawCloudlessExolabDataset(bands=["B12", "B8", "B4"], resolution=60, load_into_memory=False, transforms=transforms)
class_weights = dataset.get_class_weights()
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))
train_dataset.dataset.train()
val_dataset.dataset.eval()
test_dataset.dataset.eval()
# print("Train dataset augmented:", train_dataset.dataset.transforms.train)
# exit()
BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
model = SegNetLite(in_channels=3, out_classes=3, class_weights=class_weights)
print(model.criterion.weight)
trainer = pl.Trainer(logger=None, max_epochs=50, accelerator="gpu", devices=1)
model = SegNetLite.load_from_checkpoint("checkpoints/epoch=49-step=100.ckpt")
# trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(model, test_dataset)

def save_predictions(dataset: S2Dataset, model: torch.nn.Module, num_images: int, folder: str):
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
save_predictions(test_dataset, model, 10, "test")
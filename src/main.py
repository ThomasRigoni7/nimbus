# Trains a model using active learning: the network trains on an initial guess of the labels given by the ExoLabs and s2cloudless classification
# and later based on the training loss we select samples for which the network is most unsure and re-label them by hand (possibly increasing their weight).
import argparse
import numpy as np
import torch
from segnet import SegNetLite, SegnetWithIndex
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from datasets import dataLoaderDataset, ActiveLearningDataset
from pathlib import Path
import os
import yaml

import matplotlib.pyplot as plt

from data_augmentation import SegmentationTransforms

IN_CHANNELS = 5
OUT_CLASSES = 4
DROPOUT_PROB = 0.1

def parse_arguments():
    parser = argparse.ArgumentParser("Trains a model using active learning: the network trains on an initial guess of the labels given by the ExoLabs and s2cloudless classification and later based on the training loss we select samples for which the network is most unsure and re-label them by hand (possibly increasing their weight).")
    
    args = parser.parse_args()
    return args

def load_data(resolution: int = 10) -> tuple:
    """
    Uses Dataloader multiprocessing to load the images faster
    """
    print("Loading images...")
    dataset = dataLoaderDataset(resolution)
    loader = DataLoader(dataset, batch_size=1, num_workers=os.cpu_count(), shuffle=False)
    images = []
    for batch in loader:
        images.append((batch))
    images = {key:value.squeeze(0) for d in images for key, value in d.items()}
    print("Loading labels...")
    labels = np.load(f"data/labels/cloudless_exolabs_water/{resolution}m.npz")
    labels = {id:torch.from_numpy(label).squeeze(0) for id, label in labels.items()}
    # load possible AL labels, test labels
    ALlabels = None
    test_labels = None
    # load train indexes of "well" labeled images from file
    with open("image_ids.yaml") as f:
        id_dict = yaml.safe_load(f)
    # train_ids = id_dict["train"]
    train_ids = list(images.keys())
    return images, labels, ALlabels, test_labels, train_ids

def get_loaders(images, labels, ALlabels, test_labels, train_ids, batch_size: int = 32, img_dim:int = 512, num_workers: int = 0):
    transforms = SegmentationTransforms(False, True, True)
    ALdataset = ActiveLearningDataset(images, labels, img_dim, train_ids, test_labels, 
                                      ALlabels=ALlabels, additional_layers=["altitude", "treecover"])
    train_dataset, val_dataset, test_dataset = ALdataset.get_datasets(0.8)

    for id, img, label in val_dataset:
        path = Path(f"results/AL/{id[0]}.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.imsave(path, img[:3].numpy().transpose([1, 2, 0]))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    # TODO: implement test loader
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_loader, val_loader, None

def load_models(checkpoint_path: Path = None):
    """
    Loads the models, either an ensemble of SegNets for proposing AL candidates or a single (DeepLabV3+)
    """
    # if load_ensemble:
    #     checkpoints_dir = Path("checkpoints/ALensemble/")
    #     ensemble = [SegNetLite(in_channels=IN_CHANNELS ,out_classes=OUT_CLASSES) for _ in num_base_models]
    #     if checkpoints_dir.exists():
    #         # load the most recent checkpoints
    #         dirs = list(checkpoints_dir.glob("*"))
    #         if len(dirs) == 0:
    #             return ensemble
    #         dirs.sort()
    #         last_dir = dirs[-1]
    #         checkpoint_paths = list(last_dir.glob("*"))
    #         assert len(checkpoint_paths) == len(ensemble), "Error, number of checkpoints is different from number of base models!"
    #         for segnet, path in zip(ensemble, checkpoint_paths):
    #             segnet = SegNetLite.load_from_checkpoint(path)
    #         return ensemble

    if checkpoint_path is None:
        return SegnetWithIndex(in_channels=IN_CHANNELS ,out_classes=OUT_CLASSES, dropout_prob=DROPOUT_PROB)
    else:
        return SegnetWithIndex.load_from_checkpoint(checkpoint_path)
    # 0.207, 0.74
    return SegnetWithIndex(
        kernel_sizes=[3, 3, 3, 3, 3, 3],
        down_filter_sizes=[32, 64, 128, 256, 256, 512],
        up_filter_sizes=[256, 256, 128, 64, 32, 32],
        conv_paddings=[1, 1, 1, 1, 1, 1],
        pooling_kernel_sizes=[2, 2, 2, 2, 2, 2],
        pooling_strides=[2, 2, 2, 2, 2, 2],
        in_channels=IN_CHANNELS ,out_classes=OUT_CLASSES)
    # 0.214 -> miou: 0.71

def train_test_model(model, train_dataloader, val_dataloader, test_dataloader, epochs):
    logger = WandbLogger(name=f"base segnet p={DROPOUT_PROB}", project="nimbus", save_dir="checkpoints/AL/")
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, devices=1, accelerator="gpu")
    trainer.fit(model, train_dataloader, val_dataloader)
    if test_dataloader is not None:
        trainer.test(model, test_dataloader)
    return model

def get_AL_samples(model: SegnetWithIndex, dataloader, ensemble_dim: int = 16):
    model.set_ensemble_dim(ensemble_dim)
    trainer = pl.Trainer(devices=1, accelerator="gpu", logger=None)
    predictions = trainer.predict(model, dataloader, return_predictions=True)
    print(len(predictions))
    predictions = predictions[0]
    uncertainty_masks, uncertainty_values = predictions
    sorted_values = sorted(uncertainty_values.items(), key=lambda x: x[1], reverse=True)
    print(len(sorted_values))
    results_directory = Path("results/AL/")
    results_directory.mkdir(exist_ok=True)
    for id, value in sorted_values:
        print("ID:", id)
        print("Uncertainty:", value)
        plt.imsave(results_directory / f"{id}_uncertainty.png", uncertainty_masks[id])
        # plt.imshow(uncertainty_masks[id])
        # plt.show()
    model.set_ensemble_dim(0)
    return predictions

def main(args):
    # 1) load datasets and dataloader
    # 2) load model
    # 3) start training with low number of samples:
    #   3.1) every i epochs, interactively label some samples from the net and save new labels
    #   3.2) continue training adding new samples and replacing manual labels (put different weight in sampler?)
    #   3.3) 
    images, labels, ALlabels, test_labels, train_ids = load_data(60)
    train_loader, val_loader, test_loader = get_loaders(images, labels, ALlabels, test_labels, train_ids)
    model = load_models("checkpoints/epoch=59-step=1800.ckpt")
    # model = load_models(None)
    # train_test_model(model, train_loader, val_loader, test_loader, epochs=60)
    pred = get_AL_samples(model, val_loader)



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
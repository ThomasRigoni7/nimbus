# Trains a model using active learning: the network trains on an initial guess of the labels given by the ExoLabs and s2cloudless classification
# and later based on the training loss we select samples for which the network is most unsure and re-label them by hand (possibly increasing their weight).
import argparse
import numpy as np
import torch
from models import ModelWithIndex
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from datasets import dataLoaderDataset, ActiveLearningDataset
from pathlib import Path
import os
import yaml
from tqdm import tqdm

import matplotlib.pyplot as plt

from data_augmentation import SegmentationTransforms
from load_test_data import load_test_labels

IN_CHANNELS = 15
DROPOUT_PROB = 0.10
OUT_CLASSES = 4
AL_IMAGES = 20
CUT_IMAGE_DIM = 512

def parse_arguments():
    parser = argparse.ArgumentParser("Trains a model using active learning: the network trains on an initial guess of the labels given by the ExoLabs and s2cloudless classification and later based on the training loss we select samples for which the network is most unsure and re-label them by hand (possibly increasing their weight).")
    
    parser.add_argument("--nolog", help="Disable run logging (currently on wandb).", action="store_true")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--model", type=str, required=True, help="Architecture to train", choices=["segnet", "deeplabv3", "unet", "fpn", "psp"])
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint file to load.")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size used in training", default=32)
    parser.add_argument("--res", type=int, help="resolution to load the labels", choices=[10, 20, 60], default=10)

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
    for batch in tqdm(loader):
        images.append((batch))
    images = {key:value.squeeze(0) for d in images for key, value in d.items()}
    print("Loading labels...")
    labels = np.load(f"data/labels/cloudless_exolabs_water/{resolution}m.npz")
    labels = {id:torch.from_numpy(label).squeeze(0) for id, label in labels.items()}
    # load possible AL labels, they are 512x512 images of the form date_x_y.npy
    ALlabels = {}
    ALlabels_dir = Path("data/labels/AL")
    for path in ALlabels_dir.glob("*"):
        name = path.with_suffix("").name
        date, x, y = name.split("_")
        id = f"{date}/{x}_{y}"
        ALlabels[id] = torch.from_numpy(np.load(path))
    print(f"Loaded {len(ALlabels)} AL labels!")
    # load test labels if res = 10
    if resolution == 10:
        test_labels, test_cut_ids = load_test_labels()
    else:
        test_labels = {}
        test_cut_ids = []
    test_labels = {id:torch.from_numpy(label) for id, label in test_labels.items()}
    print(f"Loaded {len(test_labels)} test labels!")
    # load train indexes of "well" labeled images from file
    with open("image_ids.yaml") as f:
        id_dict = yaml.safe_load(f)
    train_ids = id_dict["train"]
    # train_ids = list(images.keys())
    return images, labels, ALlabels, test_labels, train_ids, test_cut_ids

def get_loaders(images, labels, ALlabels, test_labels, train_ids, test_cut_ids, batch_size: int = 32, img_dim:int = CUT_IMAGE_DIM, num_workers: int = 0, res: int = 10):
    transforms = SegmentationTransforms(False, True, True)
    ALdataset = ActiveLearningDataset(images, labels, img_dim, train_ids, test_labels, input_bands="all",
                                      ALlabels=ALlabels, additional_layers=["altitude", "treecover"], transforms=transforms)
    
    # pass cut img ids to get_datasets
    # check for ids to load:
    train_val_cut_ids_path = Path(f"cut_image_ids_{res}.yaml")
    if train_val_cut_ids_path.exists():
        with open(train_val_cut_ids_path, "r") as f:
            train_val_ids = yaml.safe_load(f)
        train_cut_ids = train_val_ids["train"]
        val_cut_ids = train_val_ids["val"]
    else:
        train_cut_ids, val_cut_ids = [], []
    
    train_dataset, val_dataset, test_dataset, AL_dataset, class_weights = ALdataset.get_datasets(train_cut_ids, val_cut_ids, test_cut_ids,train_ratio=0.8)

    # save the ids used
    if train_cut_ids == [] and val_cut_ids == []:
        train_val_ids = {
            "train": train_dataset.index2id,
            "val": val_dataset.index2id
        }
        with open(train_val_cut_ids_path, "w") as f:
            yaml.safe_dump(train_val_ids, f)


    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers) if test_dataset is not None else None
    AL_loader = DataLoader(AL_dataset, batch_size, shuffle=False, num_workers=num_workers) if AL_dataset is not None else None
    return train_loader, val_loader, test_loader, AL_loader, class_weights

def load_models(model: str, checkpoint_path: Path = None, in_channels:int=IN_CHANNELS, out_classes:int=OUT_CLASSES, class_weights:torch.Tensor=None):
    """
    Loads the model specified and returns it.

    If a checkpoint path is specified, ignores all other parameters and loads the model checkpoint at that location.
    """
    if checkpoint_path is None:
        return ModelWithIndex(model, in_channels, out_classes, class_weights, model_args={}, dropout=DROPOUT_PROB)
    else:
        return ModelWithIndex.load_from_checkpoint(checkpoint_path)

def train_test_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, log:bool = True):
    if log:
        logger = WandbLogger(name=f"unet++", project="nimbus", save_dir="checkpoints/AL/")
    else:
        logger = None
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, devices=1, accelerator="gpu")
    trainer.fit(model, train_dataloader, val_dataloader)
    if test_dataloader is not None:
        trainer.test(model, test_dataloader)
    return model

def predict(model: ModelWithIndex, dataloader: DataLoader):
    print("Calculating predictions...")
    trainer = pl.Trainer(devices=1, accelerator="gpu", logger=None)
    predictions = trainer.predict(model, dataloader, return_predictions=True)
    ret = {}
    for p in predictions:
        ret.update(p)
    return ret


def get_AL_samples(model: ModelWithIndex, dataloader, ensemble_dim: int = 16):
    model.set_ensemble_dim(ensemble_dim)
    trainer = pl.Trainer(devices=1, accelerator="gpu", logger=None)
    predictions = trainer.predict(model, dataloader, return_predictions=True)
    uncertainty_masks = {}
    cut_uncertainty_values = {}
    for masks, values in predictions:
        uncertainty_masks.update(masks)
        cut_uncertainty_values.update(values)
    
    sorted_values = sorted(cut_uncertainty_values.items(), key=lambda x: x[1], reverse=True)
    print("Most uncertain prediction:", sorted_values[0])
    results_directory = Path("results/AL/")
    results_directory.mkdir(exist_ok=True)
    model.set_ensemble_dim(0)
    return uncertainty_masks, sorted_values

def save_AL_images(images: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], uncertainty_masks: dict[str, torch.Tensor], sorted_values: list[tuple[str, float]], predictions: dict[str, torch.Tensor]):
    AL_dir = Path("AL/last_run/")
    print(f"Saving most uncertain images in {str(AL_dir)}")
    (AL_dir / "uncertainties").mkdir(exist_ok=True, parents=True)
    (AL_dir / "images").mkdir(exist_ok=True)
    (AL_dir / "labels").mkdir(exist_ok=True)
    (AL_dir / "predictions").mkdir(exist_ok=True)
    for id, value in sorted_values[:AL_IMAGES]:
        # get id, x, y and create dirs
        full_image_id, x_y = id.split("/")
        x, y = [int(v) for v in x_y.split("_")]
        unc_file = AL_dir / (f"uncertainties/%.4f_{full_image_id}_{x_y}.npy" % value)
        # save unc. mask and predictions
        np.save(unc_file, uncertainty_masks[id].detach().numpy())
        np.save(AL_dir / f"predictions/{full_image_id}_{x_y}.npy", predictions[id].detach().numpy())
        rgb_image = np.transpose(images[full_image_id].detach().numpy() [5:2:-1], [1, 2, 0])[x:x+CUT_IMAGE_DIM, y:y+CUT_IMAGE_DIM]
        false_color_image = np.transpose(images[full_image_id].detach().numpy() [13:1:-4], [1, 2, 0])[x:x+CUT_IMAGE_DIM, y:y+CUT_IMAGE_DIM]
        np.save(AL_dir / f"images/{full_image_id}_{x_y}_rgb.npy", rgb_image)
        np.save(AL_dir / f"images/{full_image_id}_{x_y}_false_color.npy", false_color_image)
        np.save(AL_dir / f"labels/{full_image_id}_{x_y}.npy", labels[full_image_id].detach().numpy()[x:x+CUT_IMAGE_DIM, y:y+CUT_IMAGE_DIM])


def main(args):
    # 1) load datasets and dataloader
    # 2) load model
    # 3) start training with low number of samples:
    #   3.1) every i epochs, interactively label some samples from the net and save new labels
    #   3.2) continue training adding new samples and replacing manual labels (put different weight in sampler?)
    #   3.3) 
    images, labels, ALlabels, test_labels, train_ids, test_cut_ids = load_data(args.res)
    train_loader, val_loader, test_loader, AL_loader, class_weights = get_loaders(images, labels, ALlabels, test_labels, train_ids, test_cut_ids, batch_size=args.batch_size, res = args.res)
    # model = load_models("checkpoints/AL/nimbus/p7bjtmz3/checkpoints/epoch=69-step=1050.ckpt")
    model = load_models(args.model, None, class_weights=class_weights)
    train_test_model(model, train_loader, val_loader, test_loader, epochs=args.epochs, log = not args.nolog)
    uncertainty_masks, sorted_unc_values = get_AL_samples(model, AL_loader)
    predictions = predict(model, AL_loader)
    save_AL_images(images, labels, uncertainty_masks, sorted_unc_values, predictions)



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
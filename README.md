# Nimbus

semantic segmentation of Sentinel 2 satellite images.

# Installation

First recreate the conda environment:
```bash
conda create --name <env_name> --file conda_packages.txt
```
# Usage

There are three main procedures to follow to use the code in this repository: the preprocessing step comes first and is necessary for everything else.
When the images have been preprocessed (extracted), then it is possible to use the scripts provided for inference and training.

## Preprocessing

From raw Sentinel2 data as downloaded from the sentinelAPI, it is first necessary to extract the images and move them to an appropriate directory. 
From the repository folder:

```bash
python src/data_preprocessing/extract_and_move_raw.py <src_folder> <dst_folder>
```
where `src_folder` is the path to the folder containing the zipped S2 data, `dst_folder` is the destination folder in which to extract the files.

It is also necessary to download the auxilliary data for the tiles where you want to make a prediction or training: those are available as Earth Engine collections.

For segmenting S2 images the DEM altitude data and the tree canopy cover are necessary, while for training also the surface water mask is necessary to generate the labels.

DEM AW3D30

`https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V3_2`

Tree Canopy Cover

`https://developers.google.com/earth-engine/datasets/catalog/NASA_MEASURES_GFCC_TC_v3`

JRC suface water

`https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater`

after downloading, run the script `src/additional_data.py` to convert all masks into 10m resolution and cut them appropriately.

## Generate Segmentation Masks

Given a pre-trained model and S2 images it is easy to generate segmentation masks:

```bash
python src/segment.py -model <model_path> -img_dir <image_directory> -out <out_path> -tile <tile_code>
```

this will automatically load the selected model and run inference on the given image with smooth tiled prediction windows. `tile_code` must be consistent with the additional data tiles processed with `src/additional_data.py`.

## Training

**WARNING: the training script loads all images and labels into memory at the same time, this means it has a huge RAM memory usage.**

The training process requires multiple steps:
 - downloading images and label data
 - generating raw image arrays
 - generating labels
 - running the main training loop script `src/main.py`
 - Optionally enter Active learning training loop by:
   - setting appropriate values to the constants at the top of `src/main.py`, optionally generate AL samples to manually segment.
   - manually segment the images with `src/image_visualizer.py`
   - move the newly generated labels into the `data/labels/AL` folder and restart `src/main.py` with the appropriate parameters.



### Download
Labels for this project were derived from the ExoLabs classification (and optionally the S2 cloudless classification) in conjunction with the aforementioned surface water mask. These can be downloaded from the Earth Engine via:


S2Cloudless
`https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY`



### Generate image arrays
To improve the image loading speed, the images are saved into numpy array files. First set your own default directories in the __init__ signature in `src/datasets/s2rawdata.py`: `dataset_dir` is the directory where the arrays will be saved, while `data_dir` (optimally a sub-directory) is the folder containing the pre-processed image data folders.

```bash
python src/generate_numpy_arrays.py
```
will generate the numpy arrays at 10m resolution and save them into the `<dataset_dir>/arrays10m` folder.

### Generate Labels

run:

```bash
python src/generate_labels.py <exolabs_folder> <dst_folder> -tile <tile_code>
```

or

```bash
python src/generate_labels.py <exolabs_folder> <dst_folder> -tile <tile_code> -cloudless_folder <cloudless_folder>
```
to include the cloudless images into the labels computation.

The `-masked` flag is useful to compare the labels to other segmentations, but will generate runtime errors when used in training. Since the labels used in this project contained very few invalid pixels, these have been ignored. `src/datasets/AL_dataset.py` contains code to adapt for this case.

The `src/generate_labels.py` script will generate a `<resolution>m.npz` file into `dst_folder`.

### Training Loop

The training loop is managed by `src/main.py`:

the `src/main.py` script will automatically load the `image_ids.yaml` and `cut_image_ids_<res>.yaml` containing respectively the full image train/validation ids (opposed to the ones in the Active Learning and test set) and the cut sample ids of the train/validation splits (512 pixels wide). If you are not using images from the 32TNS tile in 2022, then you may want to delete these files and generate your owns. `image_ids.yaml` should contain the ids (dates) of images used for training and validation purposes, while `cut_image_ids_<res>.yaml` contains the full list of cut image ids present in the train and validation sets; if not present these will be automatically generated at random with a 80/20% train/validation proportion. 

```bash
python src/main.py --model <model architecture>
```
will automatically initialize a new model with the specified architecture (one between `segnet`, `deeplabv3`, `unet`, `fpn`, `psp`), load all the images and labels into memory, train and test the model, compute the most uncertain samples and save them together with label, prediction and uncertainty mask in the `AL/last_run` folder.

### Note on pre-trained models

The models accept as input 15 bands in this order: 
 - B01
 - B02
 - B03
 - B04
 - B05
 - B06
 - B07
 - B06
 - B09
 - B10
 - B11
 - B12
 - B8A
 - DEM altitude
 - Tree canopy cover

Most models are trained with threshold scaling pre-processing method and must be also used during inference with the same method.
The images used to train the models use the processing baseline 4, which is different from the one used on the most recent S2 images (from the beginning of 2022). It has been confirmed empirically that the new processing baseline is incompatible with the old one and predictions on images with different processing methods are extremely inaccurate.
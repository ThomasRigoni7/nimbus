from pathlib import Path
from segment import predict_and_save_segmentation
from convert_raw2tif import convert_raw2tif
from save_label_tif import save_label_tif
import shutil

dates = {
 "32TNS": ["20210111T102309", "20211227T102339"],
 "13TDE": ["20210426T174859", "20211008T175241", "20211222T175739"],
 "07VEH": ["20210206T204629", "20210209T205619", "20210321T205139", "20211007T205339"],
 "32VMP": ["20210307T110819", "20210503T105619", "20211126T105309", "20211206T105329"]
}

raw_location = {
    "32TNS": Path("data/raw/raw_processed"),
    "13TDE": Path("data/new_data/13TDE"),
    "07VEH": Path("data/new_data/07VEH"),
    "32VMP": Path("data/new_data/32VMP")
}

main_folder = Path("comparisons/")
main_folder.mkdir(exist_ok=True)

for tile in dates:
    tile_folder = main_folder / tile
    for image in dates[tile]:
        image_folder = tile_folder / image
        image_folder.mkdir(exist_ok=True, parents=True)
        # create tci, fci
        convert_raw2tif(raw_location[tile] / image, image_folder/f"image_tci.tif", "rgb", True)
        convert_raw2tif(raw_location[tile] / image, image_folder/f"image_fci.tif", "fci", True)
        
        # copy labels and predictions
        save_label_tif(tile, raw_location[tile] / image, image_folder/"automatic_labels.tif")
        shutil.copy(f"predictions/fpn_5/{tile}/{image}.tif", image_folder/f"pred_fcn5.tif")

        # generate predictions for fpn_0, fpn_finetune and deeplab
        predict_and_save_segmentation([raw_location[tile] / image], "checkpoints/AL/final/fpn_0.ckpt", [image_folder/"pred_fcn0.tif"],10, 512)
        predict_and_save_segmentation([raw_location[tile] / image], "checkpoints/AL/final/fpn_6_finetune.ckpt", [image_folder/"pred_fcn_finetune.tif"],10, 512)
        predict_and_save_segmentation([raw_location[tile] / image], "checkpoints/AL/final/deeplabAL.ckpt", [image_folder/"pred_deeplabAL.tif"],10, 512)
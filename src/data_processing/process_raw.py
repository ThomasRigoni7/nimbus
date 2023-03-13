# Copy the full 1C data + cloud/snow probabilities into a single folder for further processing

from pathlib import Path
from tqdm import tqdm
import shutil


extracted = Path("data/raw/raw_extracted/")
processed = Path("data/raw/raw_processed/")
levels = ["raw_data_32TNS_1C", "raw_data_32TNS_2A"]

def write_to_single_folder(channels: list[Path], snow_mask: Path, cloud_mask: Path, dst_folder: Path):
    # print(snow_mask)
    # print(cloud_mask)
    dst_folder.mkdir(parents=True, exist_ok=True)
    for channel_file in channels:
        shutil.copy(channel_file, dst_folder / channel_file.name)
    shutil.copy(snow_mask, dst_folder / snow_mask.name)
    shutil.copy(cloud_mask, dst_folder / cloud_mask.name)




img_folders = list((extracted / levels[0]).glob("*"))
timestamps = [p.name[:26] for p in img_folders]
timestamps.sort()
timestamps = list(dict.fromkeys(timestamps))
print(timestamps)

for timestamp in tqdm(timestamps):
    # print(timestamp)
    images1C = list((extracted / levels[0]).glob(f"{timestamp}*"))
    images2A = list((extracted / levels[1]).glob(f"{timestamp.replace('1C', '2A')}*"))
    suffix = ""
    for j, (im1C, im2A) in enumerate(zip(images1C, images2A)):
        # there can be multiple images with the same timestamp
        img_data_folder = next((im1C / "GRANULE").glob("*")) / "IMG_DATA"
        mask_data_folder = next((im2A / "GRANULE").glob("*")) / "QI_DATA"
        channel_paths = list(img_data_folder.glob("*B?*.jp2"))
        cloud_mask_path = mask_data_folder / "MSK_CLDPRB_20m.jp2"
        snow_mask_path = mask_data_folder / "MSK_SNWPRB_20m.jp2"
        if len(images1C) > 1:
            suffix = f"_{j}"
        write_to_single_folder(channel_paths, snow_mask_path, cloud_mask_path, processed / (f"{timestamp[11:]}" + suffix))


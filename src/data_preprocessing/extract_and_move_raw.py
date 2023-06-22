import zipfile
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import shutil

parser = ArgumentParser(description="Extracts the Sentinel2 L1C image files from the zip format they get downloaded into a single folder per image.")
parser.add_argument("src_folder", type=str, help="folder containing multiple zip files from S2 data download.")
parser.add_argument("dst_folder", type=str, help="folder which will contain the extracted files.")

args = parser.parse_args()

original_folder = Path(args.src_folder)
extracted_folder = Path(args.dst_folder)
tmp = Path("/tmp/nimbus_extract/")
extracted_folder.mkdir(exist_ok=True, parents=True)
tmp.mkdir(exist_ok=True, parents=True)

for zip_file in tqdm(list((original_folder).glob("*.zip"))):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        tmp_folder = (tmp / zip_file.name).with_suffix("")
        namelist = zip_ref.namelist()
        images = [name for name in namelist if "IMG_DATA" in name and name.endswith(".jp2") and "TCI" not in name]
        zip_ref.extractall(tmp_folder, members=images)
        files = list(tmp_folder.glob("**/*.jp2"))
        date = zip_file.name.split("_")[2]
        (extracted_folder/date).mkdir(exist_ok=True, parents=True)
        for file in files:
            shutil.move(file, extracted_folder/date/file.name)
        shutil.rmtree(tmp_folder)

tmp.rmdir()
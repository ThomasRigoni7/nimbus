import zipfile
from pathlib import Path
from tqdm import tqdm

original_folder = Path("data/raw/")
extracted_folder = Path("data/raw/raw_extracted/")

folders_with_zip = ["raw_data_32TNS_1C", "raw_data_32TNS_2A"]

for zip_folder in folders_with_zip:
    for file in tqdm((original_folder / zip_folder).glob("*.zip")):
        (extracted_folder / zip_folder).mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder / zip_folder)
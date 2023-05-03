import numpy as np
import rasterio
from pathlib import Path
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

TEST_PATCHES_DIM = 512

def _convert_indexes(label: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """
    Converts the label indexes from the ones Cyrill used:
    0 : background -> 1
    1 : snow -> 2
    2 : dense clouds -> 0
    3 : water -> 3
    4 : semi-transparent clouds -> 0

    The final labels are:
    0 : clouds
    1 : no-snow
    2 : snow
    3 : water
    """
    new_label = np.zeros_like(label)
    new_label[label == 0] = 1
    new_label[label == 1] = 2
    new_label[label == 3] = 3
    new_label[np.logical_and(label == 2, label == 4)] = 0
    new_label[coverage] = 255
    return new_label


def _extract_regions(mask_coverage: np.ndarray) -> list[tuple[int, int]]:
    max_sum = TEST_PATCHES_DIM ** 2
    size = mask_coverage.shape[0]
    assert size == mask_coverage.shape[1]
    ret = []
    for x in range(0, 10980, 32):
        for y in range(0, 10980, 32):
            if mask_coverage[x, y] == True:
                sum = np.sum(mask_coverage[x: x+TEST_PATCHES_DIM, y:y+TEST_PATCHES_DIM])
                if sum == max_sum:
                    ret.append((x, y))
                    mask_coverage[x: x+TEST_PATCHES_DIM, y:y+TEST_PATCHES_DIM] = False
    return ret

def load_test_labels(dir: str| Path = Path("data/labels/test")):
    """
    Loads the test labels (converting the classes) and returns the labels and a list of cut image ids corresponding to their coverage.
    """
    dir = Path(dir)
    if dir.exists() == False:
        print("WARNING: test labels dir does not exist!")

    image_ids = list(dir.glob("*"))
    full_img_test_labels = {}
    cut_img_ids = []
    for id in tqdm(image_ids):
        with rasterio.open(str(id / "mask.jp2")) as mask_db:
            label = mask_db.read(1)
        with rasterio.open(str(id / "mask_coverage.jp2")) as coverage_db:
            coverage = coverage_db.read(1)

        label = _convert_indexes(label, coverage)
        regions = _extract_regions(coverage)

        full_img_test_labels[id.name] = label
        for x, y in regions:
            cut_img_ids.append(f"{id.name}/{x}_{y}")

    return full_img_test_labels, cut_img_ids

if __name__ == "__main__":
    labels, ids = load_test_labels()
    print(ids)
    keys = list(labels.keys())
    print(keys)
    print(labels[keys[0]].shape)

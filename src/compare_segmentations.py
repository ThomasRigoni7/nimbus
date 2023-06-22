import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gdal
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import argparse
import yaml

from sklearn.metrics import f1_score, jaccard_score, accuracy_score, confusion_matrix

NUM_CLASSES = 4
IMG_SIZE = 10980

def compute_seg_diff(pred: np.ndarray, label: np.ndarray):
    """
    Computes the segmentation difference between prediction and label.
    """
    assert pred.shape == label.shape
    out = np.ones_like(pred, dtype=np.uint8) * 255
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            mask = (pred == i) & (label == j)
            out[mask] = i * NUM_CLASSES + j

    return out

def get_nodata_mask(pred, label):
    return np.logical_not(np.logical_or(pred == 255, label == 255))

def compute_F1_score(pred: np.ndarray, label: np.ndarray):
    mask = get_nodata_mask(pred, label)
    return float(f1_score(label[mask], pred[mask], average="weighted"))

def compute_miou(pred, label):
    mask = get_nodata_mask(pred, label)
    return float(jaccard_score(label[mask], pred[mask], average="weighted"))

def compute_accuracy(pred, label):
    mask = get_nodata_mask(pred, label)
    return float(accuracy_score(label[mask], pred[mask]))

def save_color_diff(difference: np.ndarray, path_to_save: Path | str, transform=None, projection=None):
    CLOUD = (255, 0, 0)
    GROUND = (112, 153, 89)
    SNOW = (115, 191, 250)
    WATER = (242, 238, 162)
    colors = [CLOUD, GROUND, SNOW, WATER]

    # set the default tile as 32TNS if not specified
    if transform == None:
        transform = (499980.0, 10.0, 0.0, 5200020.0, 0.0, -10.0)
    if projection == None:
        projection = '''PROJCS["WGS 84 / UTM zone 32N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32632"]]'''

    path_to_save = Path(path_to_save)
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(str(path_to_save), IMG_SIZE, IMG_SIZE, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    outdata.SetGeoTransform(transform)
    outdata.SetProjection(projection)
    out_band = outdata.GetRasterBand(1)
    # set color interpretation
    color_table = gdal.ColorTable()
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            new_color = (np.array(colors[i]) + np.array(colors[j])) // 2
            new_colors = (new_color[0], new_color[1], new_color[2])
            color_table.SetColorEntry(i*NUM_CLASSES + j, new_colors)
    
    out_band.SetRasterColorTable(color_table)
    out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    out_band.WriteArray(difference)
    out_band.SetNoDataValue(255)
    outdata.FlushCache() ##saves to disk!!
    outdata = None

def save_confusion_matrix(preds: np.ndarray, labels: np.ndarray, title: str, path:str | Path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    conf_matrix = confusion_matrix(labels.flatten(), preds.flatten(), normalize="true", labels=[0, 1, 2, 3])

    fig, ax = plt.subplots()
    ax.imshow(conf_matrix)

    # Show all ticks and label them with the respective list entries
    classes = ["cloud", "ground", "snow", "water"]
    ax.set_xticks(np.arange(NUM_CLASSES), labels=classes)
    ax.set_yticks(np.arange(NUM_CLASSES), labels=classes)
    ax.set_xlabel("predictions")
    ax.set_ylabel("labels")
    ax.set_title(title)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    

    # Loop over data dimensions and create text annotations.
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text = ax.text(j, i, str(conf_matrix[i, j])[:4],
                            ha="center", va="center", color="w")
    plt.tight_layout()
    fig.savefig(path)
    plt.close()

def save_year_figure(vals: dict[str, float], path: str, title:str):
    vals_year = np.zeros((12, 31))
    for image_id, unc_value in vals.items():
        month = int(image_id[4:6]) - 1
        day = int(image_id[6:8]) - 1
        vals_year[month][day] += unc_value

    ax = sns.heatmap(vals_year)
    ax.set(xlabel="Day of the month", ylabel="Month")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
def load_preds(folder:str|Path):
    """
    Load predictions from the specified folder containing tif images.
    """
    folder = Path(folder)
    paths = list(folder.glob("*.tif"))
    preds = {}
    transform = None
    projection = None
    for i, p in enumerate(paths):
        ds = gdal.Open(str(p), gdal.GA_Update)
        if i == 0:
            transform = ds.GetGeoTransform()
            projection = ds.GetProjection()
        pred_band = ds.GetRasterBand(1)
        pred = pred_band.ReadAsArray()
        preds[p.with_suffix("").name] = pred
        ds = None
        pred_band = None
    return preds, transform, projection

def load_labels(path:str|Path):
    """
    Load npz file containing the labels.
    """
    labels = np.load(path)
    return labels
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script used to segment a full raw image from L1C satellite images.")
    parser.add_argument("pred_dir", type=str)
    parser.add_argument("labels_path", type=str)
    parser.add_argument("-out_dir", type=str, default="prediction_differences/last_comparison/")
    args = parser.parse_args()

    preds, transform, projection = load_preds(args.pred_dir)
    labels = load_labels(args.labels_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    global_stats = {}

    for id in tqdm(list(preds.keys())):
        pred = preds[id]
        label = labels[id[:8]]

        diff = compute_seg_diff(pred, label)
        save_color_diff(diff, path_to_save=out_dir / f"differences/{id}.tif", transform=transform, projection=projection)

        stats = {}
        stats["f1"] = compute_F1_score(pred, label)
        stats["miou"] = compute_miou(pred, label)
        stats["acc"] = compute_accuracy(pred, label)
        global_stats[id] = stats
        stats_file_path = out_dir/"stats"/f"{id}.yaml"
        stats_file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(stats_file_path, "w") as f:
            yaml.safe_dump(stats, f)
        save_confusion_matrix(pred, label, id, out_dir/f"confusion matrices/{id}.png")
    
    # swap order of dictionary
    global_stats = {"f1":{id:global_stats[id]["f1"] for id in global_stats}, 
                    "miou":{id: global_stats[id]["miou"] for id in global_stats}, 
                    "acc":{id: global_stats[id]["acc"] for id in global_stats} }
    save_year_figure(global_stats["f1"], out_dir/"f1.png", "F1")
    save_year_figure(global_stats["miou"], out_dir/"miou.png", "MIoU")
    save_year_figure(global_stats["acc"], out_dir/"accuracy.png", "Accuracy")
    print("DONE")


"""
Visualize and manually modify segmentation for a full image with matplotlib
"""
import argparse
import torch
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import PolygonSelector, LassoSelector, Slider, RadioButtons, Button, TextBox
from matplotlib.path import Path
from torchvision.utils import draw_segmentation_masks
from skimage import data, exposure, color
from skimage.segmentation import flood
import cv2

IMAGE_ORIGINAL_SIZE = 10980

class ImageVisualizer(object):
    """
    Visualizes an image in matplotlib and allows to see the segmentation mask and modify it.
    - img: np.ndarray of size [height, width, 3] (RGB image/false color image)
    """
    def __init__(self, image_ax: Axes, img: np.ndarray, label_mask: np.ndarray, uncertainty_mask: np.ndarray, prediction_mask: np.ndarray, n_labels: int, image_id: str, res: int):
        self.image_id = image_id
        self.image_ax = image_ax
        self.fig = image_ax.figure
        self.canvas = image_ax.figure.canvas
        self.selector = None

        # resample images with given resolution
        self.res = res
        self.image_size = IMAGE_ORIGINAL_SIZE
        if res != 10:
            self.image_size = (IMAGE_ORIGINAL_SIZE * 10) // res
            print(self.image_size)
            img = cv2.resize(img, (self.image_size, self.image_size))
            label_mask = cv2.resize(label_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            uncertainty_mask = cv2.resize(uncertainty_mask, (self.image_size, self.image_size))
            prediction_mask = cv2.resize(prediction_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # original RGB image in uint8 dtype
        self.full_img_rgb = (img * 255).astype(np.uint8)
        self.full_label_mask = label_mask
        self.full_uncertainty_mask = uncertainty_mask
        self.full_prediction_mask = prediction_mask

        self.x = 0
        self.y = 0
        self.dim = min(1024, self.full_label_mask.shape[0])
        self.crop_full_image()

        self.visualized = self.img.copy()
        self.path_vertices = None
        self.selection = None
        self.selection_tolerance = 10
        self.n_labels = n_labels
        self.show_segmentation = False
        self.show_predictions = False
        self.show_rgb = True
        self.add_widgets()

    def add_widgets(self):
        self.canvas.mpl_connect("key_press_event", self.on_key_pressed)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.subplots_adjust(left=0.3)
        self.image_ax.set_title("Press number to assign class to selection. 0-red-cloud, 1-green-nosnow, 2-blue-snow")

        axgamma = self.fig.add_axes([0.05, 0.25, 0.2, 0.03])
        self.gamma_slider = Slider(
            ax=axgamma,
            label="Gamma",
            valmin=0.25,
            valmax=2,
            valinit=1,
            orientation="horizontal"
        )
        self.gamma_slider.on_changed(self.set_gamma)

        axtolerance = self.fig.add_axes([0.05, 0.30, 0.2, 0.03])
        self.tolerance_slider = Slider(
            ax=axtolerance,
            label="Tolerance",
            valmin=0,
            valmax=20,
            valinit=10,
            orientation="horizontal"
        )
        self.tolerance_slider.on_changed(self.set_tolerance)

        selection_radio_ax = fig.add_axes([0.05, 0.7, 0.2, 0.2], facecolor='lightgoldenrodyellow')
        selection_radio_ax.set_title("Selection type")
        self.radio_buttons = RadioButtons(selection_radio_ax, ('flood', 'lasso', 'polygon'))
        self.radio_buttons.on_clicked(self.change_selector)

        self.transform_from = 0
        self.transform_to = 0
        def set_transform_from(x):
            self.transform_from = int(x[0])
        def set_transform_to(x):
            self.transform_to = int(x[0])
        
        transformation_from_ax = fig.add_axes([0.07, 0.55, 0.07, 0.1], facecolor='lightgoldenrodyellow')
        transformation_from_ax.set_title("Transform from:")
        self.trans_from_radio = RadioButtons(transformation_from_ax, ("0 - cloud", "1 - ground", "2 - snow", "3 - water"))
        self.trans_from_radio.on_clicked(set_transform_from)

        transformation_to_ax = fig.add_axes([0.16, 0.55, 0.07, 0.1], facecolor='lightgoldenrodyellow')
        transformation_to_ax.set_title("to:")
        self.trans_to_radio = RadioButtons(transformation_to_ax, ("0 - cloud", "1 - ground", "2 - snow", "3 - water"))
        self.trans_to_radio.on_clicked(set_transform_to)

        transform_button_ax = fig.add_axes([0.1, 0.45, 0.1, 0.05], facecolor='lightgoldenrodyellow')
        self.transform_button = Button(transform_button_ax, "transform")
        self.transform_button.on_clicked(lambda x: self.convert_selection(self.transform_from, self.transform_to))

        mode_label_ax = fig.add_axes([0.1, 0.15, 0.1, 0.05], facecolor='lightgoldenrodyellow')
        self.mode_label = TextBox(mode_label_ax, "visualized:")
        self.mode_label.set_val("image rgb")
        self.mode_label.set_active(False)

        position_label_ax = fig.add_axes([0.1, 0.1, 0.1, 0.05], facecolor='lightgoldenrodyellow')
        self.position_label = TextBox(position_label_ax, "Position (10m):")
        self.position_label.set_val("x:0, y:0")
        self.position_label.set_active(False)

        self.update_view_plain()

    def update_crop(self, key:str):
        if key == "left":
            self.y -= 1000
        if key == "right":
            self.y += 1000
        if key == "up":
            self.x -= 1000
        if key == "down":
            self.x += 1000
        self.x = min(max(self.x, 0), self.image_size - 1024)
        self.y = min(max(self.y, 0), self.image_size - 1024)
        self.position_label.set_val(f"x:{self.x * self.res // 10}, y:{self.y * self.res // 10}")
        self.crop_full_image()
        self.set_gamma(self.gamma_slider.val)

    def crop_full_image(self):
        self.img = self.full_img_rgb[self.x:self.x+self.dim, self.y:self.y+self.dim]
        self.visualized = self.img.copy()
        self.label_mask = self.full_label_mask[self.x:self.x+self.dim, self.y:self.y+self.dim]
        self.uncertainty_mask = self.full_uncertainty_mask[self.x:self.x+self.dim, self.y:self.y+self.dim]
        self.prediction_mask = self.full_prediction_mask[self.x:self.x+self.dim, self.y:self.y+self.dim]

    def update_segmentation_and_save(self):
        self.full_label_mask[self.x:self.x+self.dim, self.y:self.y+self.dim] = self.label_mask
        self.save_labels()

    def convert_selection(self, src: int, dst: int):
        if self.selection is not None and self.show_segmentation:
            mask = self.prediction_mask if self.show_predictions else self.label_mask
            mask = np.logical_and(mask == src, self.selection)
            self.label_mask[mask] = dst
            if self.show_predictions:
                self.prediction_mask[mask] = dst
            self.update_segmentation_and_save()
        self.show_predictions = False
        self.update_view_segmentation()

    def set_gamma(self, gamma_value: float):
        self.visualized = exposure.adjust_gamma(self.img, gamma_value)
        self.update_view()

    def set_tolerance(self, value):
        self.selection_tolerance = value

    def change_selector(self, label):
        print(label)
        if label == "flood":
            self.use_flood_selector()
        elif label == "lasso":
            self.use_lasso_selector()
        elif label == "polygon":
            self.use_polygon_selector()

    def use_polygon_selector(self):
        if self.selector is not None:
            self.disconnect_selector()
        self.selector = PolygonSelector(self.image_ax, onselect=self.onselect)

    def use_lasso_selector(self):
        if self.selector is not None:
            self.disconnect_selector()
        self.selector = LassoSelector(self.image_ax, onselect=self.onselect)

    def use_flood_selector(self):
        if self.selector is not None:
            self.disconnect_selector()
        self.selector = None

    def disconnect_selector(self):
        self.selector.disconnect_events()

    def np2torch(self, img:np.ndarray):
        return torch.from_numpy(img.transpose([2, 0, 1]))
    
    def torch2np(self, img:torch.tensor):
        return img.numpy().transpose(1, 2, 0)

    def convert_masks_to_bool_multichannel(self):
        if self.show_predictions:
            mask = self.prediction_mask
        else:
            mask = self.label_mask
        masks = mask==np.arange(self.n_labels)[:, None, None]
        return torch.from_numpy(masks)
     
    def update_view_segmentation(self):
        img = self.np2torch(self.visualized)
        img = draw_segmentation_masks(img, masks=self.convert_masks_to_bool_multichannel(), colors=["red", "green", "blue", "lightyellow"], alpha=0.3)
        self.image_ax.imshow(self.torch2np(img))
        self.mode_label.set_val("predictions" if self.show_predictions else "labels")
        self.canvas.draw_idle()
        self.show_segmentation = True

    def update_view_plain(self):
        self.image_ax.imshow(self.visualized)
        self.mode_label.set_val("image rgb" if self.show_rgb else "image false color")
        self.canvas.draw_idle()
        self.show_segmentation = False
    
    def switch_view(self):
        self.show_segmentation = not self.show_segmentation
        self.update_view()

    def switch_label_prediction(self):
        self.show_predictions = not self.show_predictions

    def update_view(self):
        if self.show_segmentation:
            self.update_view_segmentation()
        else:
            self.update_view_plain()
        if self.selection is not None:
            self.image_ax.imshow(self.selection, cmap='gray', alpha=0.2)
        self.canvas.draw_idle()
        
    def update_view_uncertainties(self):
        self.image_ax.imshow(self.uncertainty_mask)
        self.mode_label.set_val("uncertainties")
        self.canvas.draw_idle()

    def update_segmentation_mask(self, label: int):
        if self.selection is None:
            return
        self.label_mask[self.selection] = label
        self.update_segmentation_and_save()
        self.show_segmentation = True
        self.show_predictions = False
        self.update_view()

    def on_key_pressed(self, event):
        key:str = event.key
        if key.isnumeric():
            self.update_segmentation_mask(int(key))
        if key == " ":
            self.switch_view()
        if key == "escape":
            self.selection = None
            self.update_view()
        if key == "u":
            self.update_view_plain()
            self.update_view_uncertainties()
        if key == "p":
            self.switch_label_prediction()
            self.update_view_segmentation()
        if key == "right" or key == "left" or key == "up" or key == "down":
            self.update_crop(key)

    def onselect(self, verts):
        init = np.array(verts)
        xv, yv = init[:, 0], init[:, 1]
        init = np.stack([yv, xv]).T
        self.path_vertices = np.concatenate([init, init[0][None, :]], axis=0)
        nr, nc = self.img.shape[0], self.img.shape[1]
        xgrid, ygrid = np.mgrid[:nr, :nc]
        xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
        path = Path(self.path_vertices, closed=True)
        mask = path.contains_points(xypix)
        mask = mask.reshape((nr, nc))
        if self.selection is not None:
            self.selection = np.logical_and(self.selection, mask)
        else:
            self.selection = mask
        self.update_view()

    def flood_select(self, pos):
        """
        Uses flood algorithm to select pixels, from the grayscale image.
        """
        self.selection = flood(color.rgb2gray(self.img), pos, tolerance=self.selection_tolerance/100)
        print("Selected pixels:", np.count_nonzero(self.selection))
        print("Sum:", self.selection.sum())
        self.update_view()

    def on_mouse_press(self, event):
        x, y = event.xdata, event.ydata
        if event.inaxes is self.image_ax and x is not None and y is not None:
            x = int(x+0.49)
            y = int(y+0.49)
            if self.selector is None:
                self.flood_select((y, x))

    def save_labels(self):
        path = f"output/{self.image_id}.npy"
        # interpolate if not full res
        if self.res != 10:
            labels = cv2.resize(self.full_label_mask, (IMAGE_ORIGINAL_SIZE, IMAGE_ORIGINAL_SIZE), interpolation=cv2.INTER_NEAREST)
            np.save(path, labels)
        else:
            np.save(path, self.full_label_mask)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_id", nargs="*", type=str)
    parser.add_argument("-dir", type=str, default="AL")
    parser.add_argument("-res", type=int, choices=[10, 20, 30, 40, 50, 60, 100], default=10)
    parser.add_argument("-rgb", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    if not pathlib.Path(args.dir).exists():
        raise ValueError("Specified directory does not exist!")

    if args.image_id == []:
        files = list((pathlib.Path(args.dir) / "labels").glob("*"))
        args.image_id = [f.with_suffix("").name for f in files]

    print(args.image_id)

    for id in args.image_id:
        fig, ax = plt.subplots(figsize = (15, 10))
        if args.rgb:
            img = np.load(pathlib.Path(args.dir) / f"images/{id}_rgb.npy")
        else:
            img = np.load(pathlib.Path(args.dir) / f"images/{id}_false_color.npy")
        manual_label_path = pathlib.Path(f"output/{id}.npy")
        if manual_label_path.exists():
            res = input("Manual label found. do you want to load it (y/n)? (y):")
            if res != "n":
                print("loading manual label")
                label = np.load(manual_label_path)
            else:
                print("loading regular label")
                label = np.load(pathlib.Path(args.dir) / f"labels/{id}.npy")
        else:
            print("loading regular label")
            label = np.load(pathlib.Path(args.dir) / f"labels/{id}.npy")
        unc_list = list((pathlib.Path(args.dir) / f"uncertainties/{id}").glob("*"))
        uncertainties = np.zeros_like(label, dtype=np.float32)
        for unc_path in unc_list:
            h_w = unc_path.name
            h, w, _ = h_w.split("_")
            h, w = int(h), int(w)
            unc = np.load(unc_path)
            uncertainties[h:h+unc.shape[0], w:w+unc.shape[1]] = unc
        pred_list = list((pathlib.Path(args.dir) / f"predictions/{id}").glob("*"))
        predictions = np.zeros_like(label)
        for pred_path in pred_list:
            h_w = pred_path.with_suffix("").name
            h, w = h_w.split("_")
            h, w = int(h), int(w)
            pred = np.load(pred_path)
            predictions[h:h+pred.shape[0], w:w+pred.shape[1]] = pred

        visualizer = ImageVisualizer(ax, img, label, uncertainties, predictions, int(label.max()) + 1, id, args.res)

        plt.show()

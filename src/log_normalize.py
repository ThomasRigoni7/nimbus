# calculate the log-normalization values for the images
import numpy as np
from datasets import S2RawData
import matplotlib.pyplot as plt
import argparse

res = 60

def log_normalize(image: np.ndarray, a, b, c, d):
    if len(image) == 15:
        image = image[2:]
    else:
        raise ValueError("Image must be 15 channels!")
    image = image.transpose(1, 2, 0)
    log_img = np.log(image)
    log_c = np.log(c)
    log_d = np.log(d)
    print("log c:", log_c)
    print("log d:", log_d)

    return sigmoid(((log_img - log_c) * (b - a) / (log_d - log_c)) + a)

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def inv_sigmoid(y):
    x = np.log(1/y - 1)
    return x

def calculate_values(images_dict, lower_quantile, higher_quantile):
    print("Calculating normalization values...")
    # filter out all the snow/cloud masks
    images_list = []
    for id, img in images_dict.items():
        images_list.append(img)
    images = np.stack(images_list)
    images = S2RawData.to_4_dim(images)
    if images.shape[1] == 15:
        images = images[:, 2:]
    else:
        raise ValueError()
    # a: output lower sigmoid bound (x s.t. y = lower_percentile on sigmoid)
    # b: output upper sigmoid bound (x s.t. y = higher_percentile on sigmoid)
    # c: input lower bound (lower_percentile % on input)
    # d: input upper bound (higher_percentile % on input)
    a = inv_sigmoid(lower_quantile)
    print("a:", a)
    b = inv_sigmoid(higher_quantile)
    print("b:", b)
    c, d = np.quantile(images, [lower_quantile, higher_quantile], axis=[0, 2, 3])
    print("c:", c)
    print("d:", d)
    return a, b, c, d

def visualize_image(image):

    linear_image = s2data.preprocess(image).squeeze(0)
    log_normalized_image = log_normalize(image.astype(np.float32), a, b, c, d)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(10, 9))
    print(linear_image.shape)
    ax1.imshow(linear_image[13:1:-4].transpose(1, 2, 0))
    ax2.imshow(log_normalized_image[:, :, 11:0:-4])
    ax3.imshow(linear_image[5:2:-1].transpose(1, 2, 0))
    ax4.imshow(log_normalized_image[:, :, 3:0:-1])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script used to compute the log-normalization values for the image pre-processing.")
    parser.add_argument("-data_dir", help="directory containing the images", required=True)
    parser.add_argument("-res", help="resolution", default=60)

    args = parser.parse_args()

    print("loading images...")
    s2data = S2RawData(data_dir=args.data_dir, resolution=args.res)
    images_dict = s2data.load_arrays(resolution=args.res, return_dict=True)
    a, b, c, d = calculate_values(images_dict, 0.3, 0.7)
    np.savez("log_normalization_values.npz", a=a, b=b, c=c, d=d)
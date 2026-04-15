import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.impute import KNNImputer
from skimage.color import rgb2hsv, hsv2rgb
from skimage.morphology import disk, dilation


def load_image(image_path):
    """Load image as float RGB in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img = np.asarray(img).astype(np.float64) / 255.0
    return img


def save_image(img, save_path):
    """Save float image in [0,1] to disk."""
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path)


def knn_impute_channel(channel, k):
    """
    Approximate MATLAB knnimpute for a 2D image channel.
    Treat each row as a sample and each column as a feature.
    """
    imputer = KNNImputer(n_neighbors=k)
    return imputer.fit_transform(channel)


def pre_process_hsv(X, t, k, m):
    """
    Python translation of the MATLAB preprocessing function.

    Parameters
    ----------
    X : np.ndarray
        RGB image as float array in [0,1], shape (H, W, 3)
    t : float
        Max value of the input image
    k : int
        Number of neighbors for KNN imputation
    m : float
        Max threshold of the reference image

    Returns
    -------
    imgnoword : np.ndarray
        Image after letter removal + imputation
    imgadapt : np.ndarray
        Image after threshold adaptation
    """

    # Convert RGB to HSV
    X = rgb2hsv(X)

    # Fill dark place with blue color
    dark_threshold = 0.148
    v_channel = X[:, :, 2]
    dark_mask = v_channel < dark_threshold

    blue_color = np.array([0.606, 1.000, 1.000])  # HSV

    for c in range(3):
        X[:, :, c][dark_mask] = blue_color[c]

    # Convert back to RGB
    X = hsv2rgb(X)

    # Create letter mask using HSV thresholding
    I = rgb2hsv(X)

    channel1_min, channel1_max = 0.000, 1.000
    channel2_min, channel2_max = 0.700, 1.000
    channel3_min, channel3_max = 0.000, 1.000

    BW = (
        (I[:, :, 0] >= channel1_min) & (I[:, :, 0] <= channel1_max) &
        (I[:, :, 1] >= channel2_min) & (I[:, :, 1] <= channel2_max) &
        (I[:, :, 2] >= channel3_min) & (I[:, :, 2] <= channel3_max)
    )

    # maskedRGBImage = X; maskedRGBImage(~BW)=0; not used later, omitted

    BW = BW.astype(np.float64)
    BW = 1.0 - BW  # imcomplement

    # Dilate the mask
    radius = 6
    selem = disk(radius)
    expanded_mask = dilation(BW > 0, selem)

    # Remove letters: set masked pixels to 0
    X_removed = X.copy()
    X_removed[expanded_mask] = 0

    # Split channels
    R = X_removed[:, :, 0].copy()
    G = X_removed[:, :, 1].copy()
    B = X_removed[:, :, 2].copy()

    # Find zero-valued pixels across all channels
    zero_mask = (R == 0) & (G == 0) & (B == 0)

    # Replace those positions with NaN
    R[zero_mask] = np.nan
    G[zero_mask] = np.nan
    B[zero_mask] = np.nan

    # KNN imputation for each channel
    r_imputed = knn_impute_channel(R, k)
    g_imputed = knn_impute_channel(G, k)
    b_imputed = knn_impute_channel(B, k)

    imgnoword = np.stack([r_imputed, g_imputed, b_imputed], axis=-1)
    imgnoword = np.clip(imgnoword, 0, 1)

    # Threshold customizing
    imgnoword_hsv = rgb2hsv(imgnoword)
    h = imgnoword_hsv[:, :, 0].copy()
    s = imgnoword_hsv[:, :, 1].copy()
    v = imgnoword_hsv[:, :, 2].copy()

    h[h > 0.80] = 0.001

    # MATLAB:
    # h = ((1.5*t - 0.75)*h - t + m)/((m-0.5)*1.5);
    h = ((1.5 * t - 0.75) * h - t + m) / ((m - 0.5) * 1.5)

    v[v != 1] = 1
    s[s != 1] = 1

    imgadapt_hsv = np.stack([h, s, v], axis=-1)
    imgadapt = hsv2rgb(imgadapt_hsv)
    imgadapt = np.clip(imgadapt, 0, 1)

    # MATLAB converts imgnoword back from HSV to RGB here, but in this Python
    # version imgnoword is already RGB after KNN imputation, so we keep it as is.
    imgnoword = np.clip(imgnoword, 0, 1)

    return imgnoword, imgadapt

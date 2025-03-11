import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from src.collect_data import load_data


def create_output_dir(dir_name="output_images_bayesian_blue_2"):
    """
    Create an output directory if it doesn't exist yet.

    Parameters
    ----------
    dir_name : str
        Name of the directory to be created.

    Returns
    -------
    str
        The name (path) of the created or existing directory.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def save_image(image, filename, output_dir):
    """
    Save an image as a PNG file in the specified directory.

    Parameters
    ----------
    image : np.ndarray
        The image array (either grayscale or color).
    filename : str
        The name of the file to be saved (e.g., 'result.png').
    output_dir : str
        Directory path where the image should be saved.

    Returns
    -------
    None
    """
    filepath = os.path.join(output_dir, filename)
    # If the image is color (3 channels), convert from RGB to BGR before saving.
    if len(image.shape) == 3 and image.shape[2] == 3:
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # For grayscale or mask images, save directly
        cv2.imwrite(filepath, image)


def reinhard_color_normalization(source, target_means, target_stds):
    """
    Perform color normalization using Reinhard's method.

    Parameters
    ----------
    source : np.ndarray
        The source image in BGR format.
    target_means : np.ndarray
        The target mean values (LAB) for normalization (3-element array).
    target_stds : np.ndarray
        The target standard deviations (LAB) for normalization (3-element array).

    Returns
    -------
    np.ndarray
        The color-normalized image in BGR format.
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)

    # Calculate the means and stds of the source image in LAB
    source_means = np.mean(source_lab, axis=(0, 1))
    source_stds = np.std(source_lab, axis=(0, 1))

    # Perform Reinhard normalization
    normalized_lab = source_lab.copy()
    for i in range(3):
        normalized_lab[:, :, i] = (
            (normalized_lab[:, :, i] - source_means[i])
            * (target_stds[i] / source_stds[i])
        ) + target_means[i]

    # Clip the result to valid range and convert back to BGR
    normalized_lab = np.clip(normalized_lab, 0, 255)
    return cv2.cvtColor(normalized_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def display_image(title, image, cmap=None):
    """
    Display an image using Matplotlib.

    Parameters
    ----------
    title : str
        The title of the display window (plot).
    image : np.ndarray
        The image array.
    cmap : str or None
        The colormap for grayscale images. If None, the default colormap is used for RGB images.

    Returns
    -------
    None
    """
    plt.figure()
    plt.title(title)
    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap)
    else:
        # If the image is color in BGR, convert to RGB for proper display
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def morphological_operations(image, operation, kernel_size=(5, 5), iterations=1):
    """
    Apply basic morphological operations (erosion, dilation, opening, closing) with an elliptical kernel.

    Parameters
    ----------
    image : np.ndarray
        Input binary or grayscale image.
    operation : str
        Type of morphological operation ('erosion', 'dilation', 'opening', 'closing').
    kernel_size : tuple, optional
        Size of the elliptical kernel (default is (5, 5)).
    iterations : int, optional
        Number of iterations to apply the operation (default is 1).

    Returns
    -------
    np.ndarray
        The processed image after the morphological operation.

    Raises
    ------
    ValueError
        If the operation string is not one of the supported operations.
    """
    # Create an elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # Perform the specified morphological operation
    if operation == "erosion":
        result = cv2.erode(image, kernel, iterations=iterations)
    elif operation == "dilation":
        result = cv2.dilate(image, kernel, iterations=iterations)
    elif operation == "opening":
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations)
    elif operation == "closing":
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations)
    else:
        raise ValueError(
            "Invalid operation. Choose from 'erosion', 'dilation', 'opening', or 'closing'."
        )

    return result


# Constants for morphological operations
EROSION = "erosion"
DILATION = "dilation"
OPENING = "opening"
CLOSING = "closing"


def show_anns(anns):
    """
    Display annotations (masks) generated by some segmentation algorithm,
    overlayed on the current Matplotlib axes.

    Parameters
    ----------
    anns : list
        A list of annotation dictionaries, each containing at least:
        - 'segmentation': a 2D boolean mask
        - 'area': numeric value for the area of the mask

    Returns
    -------
    None
    """
    if len(anns) == 0:
        return

    # Sort annotations by area (descending)
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        m = ann["segmentation"]  # 2D boolean mask
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def convert_white_background_to_black(img_color, threshold=250):
    """
    Converts white background to black, assuming that "white" pixels
    have all (R, G, B) values above the specified threshold.

    Parameters
    ----------
    img_color : np.ndarray
        Input color image with a potentially white background.
    threshold : int
        Threshold above which a pixel is considered "white". Default is 250.

    Returns
    -------
    img_out : np.ndarray
        The modified image with black background where the original pixels
        exceeded the threshold in all channels (R, G, B).
    """
    img_out = np.copy(img_color)
    white_mask = (
        (img_out[..., 0] > threshold)
        & (img_out[..., 1] > threshold)
        & (img_out[..., 2] > threshold)
    )
    # Set those pixels to (0, 0, 0), i.e., black
    img_out[white_mask] = [0, 0, 0]
    return img_out

def data_imputation(data):
    # Extract rows for class G1
    df_g1 = data[data["target"] == "G1"].copy()

    # Current number of samples for G1, and the desired number
    current_size = len(df_g1)       # e.g., 22
    target_size = 53                # Your target number of samples

    if current_size < target_size:
        n_needed = target_size - current_size
        
        # Separate numerical features (excluding the "target" column)
        df_g1_features = df_g1.drop(columns=["target"])
        
        # Calculate the mean and std for each column
        means = df_g1_features.mean()
        stds = df_g1_features.std()
        
        # Initialize an array to store the new oversampled rows
        new_data = np.zeros((n_needed, df_g1_features.shape[1]))
        
        # For each column in df_g1_features, generate n_needed values
        for i, col in enumerate(df_g1_features.columns):
            mu = means[col]
            sigma = stds[col] if stds[col] != 0 else 1e-9  # safeguard in case std=0
            new_data[:, i] = np.random.normal(mu, sigma, n_needed)
        
        # Create a DataFrame from the generated samples and add the G1 class label
        df_new_g1 = pd.DataFrame(new_data, columns=df_g1_features.columns)
        df_new_g1["target"] = "G1"
        
        # Concatenate the new samples to the original dataset
        data = pd.concat([data, df_new_g1], ignore_index=True)
        
        print(f"Added {n_needed} new rows for class G1.")
        print("Current class distribution:\n", data["target"].value_counts())
    else:
        print("Class G1 already has enough samples.")

    print(data.shape)
    return data



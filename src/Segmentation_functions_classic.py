import cv2
import numpy as np
import os
from skimage import morphology
from skimage.filters import threshold_minimum
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from cellpose import models
from skimage.color import label2rgb
import torch
import skfuzzy as fuzz
from utils import create_output_dir, save_image, display_image, reinhard_color_normalization, convert_white_background_to_black


def process_and_visualize_he_image_blue_channel_with_morphology(
    image_path,
    save_intermediate=False,
    desired_size=None,            # (width, height) tuple or None
    apply_median_filter=False,    # Whether to apply medianBlur
    median_kernel_size=3,         # Median kernel size
    increase_contrast=False,      # Whether to apply additional global contrast
    alpha=1.2,                    # Gain (contrast)
    beta=15,                      # Bias (brightness)
    adjust_blue_after_norm=False, # Whether to additionally adjust the blue channel after normalization
    blue_factor=1.0,              # Multiplier for the blue channel (<1 darkens, >1 brightens)
    blue_offset=0,                # Offset (positive brightens, negative darkens)
    morph_kernel_size=3           # Kernel size for morphological operations
):
    """
    Process an H&E-stained image focusing on the blue channel, applying 
    morphological operations before thresholding. This pipeline includes:

     1. Loading the image.
     2. (Optional) Resizing.
     3. (Optional) Median filter.
     4. (Optional) Additional contrast with linear transform (alpha, beta).
     5. Color normalization using Reinhard's method.
     6. (Optional) Adjusting/darkening the blue channel after normalization.
     7. Extraction of the blue channel.
     8. CLAHE for contrast enhancement.
     9. Selective blue color masking.
     10. Combining the mask with the enhanced blue channel.
     11. Morphological operations to fill holes and remove noise.
     12. Bayesian thresholding.
     13. Creation of a segmentation mask.
     14. Post-processing (removing small objects and holes).
     15. Overlaying the mask on the (possibly resized) image, converting 
         background to white.

    Parameters
    ----------
    image_path : str
        Path to the image file (e.g., .png or .jpg).
    save_intermediate : bool
        Whether to save and display intermediate steps.
    desired_size : tuple or None
        If not None (e.g., (512, 512)), the image is resized to this size.
    apply_median_filter : bool
        Whether to apply medianBlur before color normalization.
    median_kernel_size : int
        Kernel size for the median filter.
    increase_contrast : bool
        Whether to apply linear contrast/brightness adjustment (alpha, beta).
    alpha : float
        Contrast coefficient.
    beta : float
        Brightness coefficient.
    adjust_blue_after_norm : bool
        Whether to manipulate the blue channel after normalization.
    blue_factor : float
        Multiplier for the blue channel (<1 darkens, >1 brightens).
    blue_offset : int
        Offset (positive brightens, negative darkens).
    morph_kernel_size : int
        Kernel size for morphological operations (e.g., 3, 5, 7, etc.).

    Returns
    -------
    nuclei_on_white : np.ndarray
        Image with segmented nuclei overlaid on a white background.
    """
    output_dir = create_output_dir()

    # 1. Load the original image from disk
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    if save_intermediate:
        save_image(image, "1_original.png", output_dir)
        display_image("Original Image", image)

    # 2. (Optional) Resizing
    if desired_size is not None:
        image = cv2.resize(image, desired_size, interpolation=cv2.INTER_AREA)
        if save_intermediate:
            save_image(image, "1a_resized.png", output_dir)
            display_image("Resized Image", image)

    # 3. (Optional) Median Filtering before color normalization
    if apply_median_filter:
        image = cv2.medianBlur(image, median_kernel_size)
        if save_intermediate:
            save_image(image, "1b_median_filtered.png", output_dir)
            display_image("Median Filtered Image", image)

    # 4. (Optional) Additional Contrast (linear transform: out = alpha * image + beta)
    if increase_contrast:
        # convertScaleAbs applies saturating cast to uint8.
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if save_intermediate:
            save_image(image, "1c_contrast_enhanced.png", output_dir)
            display_image("Contrast Enhanced Image", image)

    # 5. Reinhard color normalization
    target_means = np.array([128, 128, 128])
    target_stds = np.array([20, 20, 20])
    normalized_image = reinhard_color_normalization(image, target_means, target_stds)
    if save_intermediate:
        save_image(normalized_image, "2_normalized.png", output_dir)
        display_image("Normalized Image", normalized_image)

    # 6. (Optional) Adjust the blue channel after normalization
    #    before extracting it. This can help if small nuclei 
    #    are too bright/dim.
    if adjust_blue_after_norm:
        # Extract the blue channel and treat it as float for manipulation
        blue_float = normalized_image[:, :, 0].astype(np.float32)
        # Apply multiplier
        blue_float *= blue_factor
        # Apply offset
        blue_float += blue_offset
        # Clip to [0, 255]
        blue_float = np.clip(blue_float, 0, 255)
        # Insert back into the image (as uint8)
        normalized_image[:, :, 0] = blue_float.astype(np.uint8)
        if save_intermediate:
            tmp_show = normalized_image[:, :, 0]
            save_image(tmp_show, "2a_adjusted_blue_channel.png", output_dir)
            display_image("Adjusted Blue Channel", tmp_show, cmap="gray")

    # 7. Extract the (potentially adjusted) blue channel
    blue_channel = normalized_image[:, :, 0]
    if save_intermediate:
        save_image(blue_channel, "3_blue_channel.png", output_dir)
        display_image("Blue Channel", blue_channel, cmap="gray")

    # 8. Apply CLAHE on the blue channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blue_channel_eq = clahe.apply(blue_channel)
    if save_intermediate:
        save_image(blue_channel_eq, "4_blue_channel_eq.png", output_dir)
        display_image("CLAHE on Blue Channel", blue_channel_eq, cmap="gray")

    # 9. Create a selective blue color mask in HSV space
    hsv_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 20])   # Adjust as needed
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    if save_intermediate:
        save_image(blue_mask, "5_blue_mask.png", output_dir)
        display_image("Blue Mask", blue_mask, cmap="gray")

    # 10. Combine (bitwise AND) the blue mask with the enhanced blue channel
    masked_blue_channel = cv2.bitwise_and(blue_channel_eq, blue_channel_eq, mask=blue_mask)
    if save_intermediate:
        save_image(masked_blue_channel, "6_masked_blue_channel.png", output_dir)
        display_image("Masked Blue Channel", masked_blue_channel, cmap="gray")

    # 11. Morphological operations before thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    morphed_channel = cv2.morphologyEx(masked_blue_channel, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed_channel = cv2.morphologyEx(morphed_channel, cv2.MORPH_OPEN, kernel, iterations=1)
    if save_intermediate:
        save_image(morphed_channel, "7_morphed_channel.png", output_dir)
        display_image("Morphological Enhancement", morphed_channel, cmap="gray")

    # 12. Bayesian thresholding (threshold_minimum from skimage)
    threshold = threshold_minimum(morphed_channel)
    print(f"Calculated Bayesian threshold: {threshold}")
    offset = 0  # Adjust if needed
    adjusted_threshold = threshold + offset
    print(f"Adjusted threshold: {adjusted_threshold}")

    # 13. Create a segmentation mask
    segmentation_mask = np.where(morphed_channel >= adjusted_threshold, 255, 0).astype(np.uint8)
    if save_intermediate:
        save_image(segmentation_mask, "9_segmentation_mask.png", output_dir)
        display_image("Segmentation Mask", segmentation_mask, cmap="gray")

    # 14. Remove small objects and small holes
    #    Reduced 'min_size' and 'area_threshold' to 50 as an example
    segmentation_mask_bool = segmentation_mask.astype(bool)
    cleaned_seg_mask = morphology.remove_small_objects(segmentation_mask_bool, min_size=50)
    cleaned_seg_mask = morphology.remove_small_holes(cleaned_seg_mask, area_threshold=50)
    cleaned_seg_mask = cleaned_seg_mask.astype(np.uint8) * 255
    if save_intermediate:
        save_image(cleaned_seg_mask, "11_cleaned_channel.png", output_dir)
        display_image("Cleaned Channel", cleaned_seg_mask, cmap="gray")

    # 15. Overlay the mask on the (possibly resized) image
    #     Convert background to white
    mask_3ch = cv2.merge([cleaned_seg_mask, cleaned_seg_mask, cleaned_seg_mask])
    nuclei_segmented = cv2.bitwise_and(image, mask_3ch)
    white_background = 255 * np.ones_like(image, dtype=np.uint8)
    nuclei_on_white = np.where(mask_3ch == 0, white_background, nuclei_segmented)

    if save_intermediate:
        save_image(nuclei_on_white, "10_nuclei_segmented_white_bg.png", output_dir)
        display_image("Segmented Nuclei on White Background", nuclei_on_white)

    return nuclei_on_white

def fuzzy_cmeans_in_rgb_2clusters(
    cell_on_black,
    convert_white_bg_to_black=True,
    threshold_for_white=250,
    save_intermediate=False,
    output_dir="output_images",
    min_size=50
):
    """
    Performs Fuzzy C-means clustering (2 clusters) in the BGR color space,
    with the option to convert a white background to black first. 
    Automatically picks the cluster with the lower sum of BGR values as nuclei.

    Parameters
    ----------
    cell_on_black : np.ndarray
        An image in BGR format where the background is either entirely black (0,0,0)
        or, if convert_white_bg_to_black=True, it may start as white (and will be converted).
    convert_white_bg_to_black : bool
        If True, attempts to convert any white background to black before clustering.
    threshold_for_white : int
        Threshold above which a pixel is considered "white" in all channels.
    save_intermediate : bool
        If True, displays (and potentially saves) intermediate steps.
    output_dir : str
        Directory where intermediate images could be saved (if the save_image function is used).
    min_size : int
        Minimum object size (in pixels) to keep after fuzzy clustering (remove_small_objects).

    Returns
    -------
    final_mask : np.ndarray
        A 2D binary mask (dtype=uint8) indicating nuclei (1) vs. background (0).
    masked_nuclei : np.ndarray
        A color image of the same shape as `cell_on_black`, where the background
        is black and the detected nuclei are shown in their original colors.
    """
     
    # STEP 0: (Optional) Convert a white background to black
     
    if convert_white_bg_to_black:
        cell_on_black = convert_white_background_to_black(cell_on_black, threshold_for_white)
        if save_intermediate:
            display_image("Converted White to Black BG", cell_on_black)
            save_image(cell_on_black, "converted_white_to_black.png", output_dir)

     
    # STEP 1: Show or save the input image (now guaranteed to be black BG)
     
    if save_intermediate:
        display_image("Original (cell_on_black)", cell_on_black)
        save_image(cell_on_black, "original_cell_on_black.png", output_dir)

    # Apply a median blur to reduce small noise
    blurred_img = cv2.medianBlur(cell_on_black, 3)

    # STEP 2: Create a mask for black background vs. cell region
     
    black_mask = np.all(blurred_img == [0, 0, 0], axis=-1)  # shape: (H, W)
    cell_mask = ~black_mask                                # True for pixels belonging to cells

    h, w, _ = cell_on_black.shape

     
    # STEP 3: Extract the pixels corresponding to the cell (B, G, R)
     
    cell_pixels_bgr = blurred_img[cell_mask]  # shape: (N, 3)
    if cell_pixels_bgr.size == 0:
        print("No cell pixels found — possibly an empty or fully black image.")
        return None, None

    # Prepare data for fuzzy clustering: we want shape (3, N)
    data = cell_pixels_bgr.T.astype(np.float32)

     
    # STEP 4: Fuzzy C-means clustering with 2 clusters
     
    c = 2
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, 
        c=c, 
        m=2.0,       # fuzziness parameter
        error=0.005, # convergence criterion
        maxiter=1000,
        init=None
    )

    print("Cluster centroids (BGR space):")
    for i in range(c):
        print(f"  Cluster {i}: {cntr[i]}")

     
    # STEP 5: Automatically pick the cluster with the lower sum of BGR values
    #         as the "nucleus" cluster
     
    cluster_sums = [np.sum(cntr[i]) for i in range(c)]
    nucleus_cluster_idx = int(np.argmin(cluster_sums))
    print(f"Choosing cluster {nucleus_cluster_idx} as the nucleus cluster (lowest sum of BGR).")

    # Assign each pixel to whichever cluster has the highest membership
    cluster_membership = np.argmax(u, axis=0)  # shape: (N,)

    # Build a 2D mask for the final segmentation
    fuzzy_mask_2d = np.zeros((h, w), dtype=np.uint8)
    fuzzy_mask_2d[cell_mask] = (cluster_membership == nucleus_cluster_idx).astype(np.uint8)

    if save_intermediate:
        display_image("Fuzzy Mask 2D (initial)", fuzzy_mask_2d * 255, cmap="gray")
        save_image(fuzzy_mask_2d * 255, "fuzzy_mask_2d_initial.png", output_dir)

     
    # STEP 6: Morphological cleaning (fill holes, remove small objects, etc.)
     
    # Fill holes
    mask_filled = binary_fill_holes(fuzzy_mask_2d).astype(np.uint8)

    # Remove small objects
    mask_cleaned = remove_small_objects(mask_filled.astype(bool), min_size=min_size).astype(np.uint8)

    # Close operation to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    if save_intermediate:
        display_image("Final Mask (morphological close)", final_mask * 255, cmap="gray")
        save_image(final_mask * 255, "final_mask_morph_close.png", output_dir)

    # (Optional) refine edges with a dilation + another closing
    mask_dilated = cv2.dilate(final_mask, kernel, iterations=1)
    final_mask = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)

     
    # STEP 7: Overlay the final mask on the original image (with black BG)
     
    masked_nuclei = np.zeros_like(cell_on_black)
    masked_nuclei[final_mask == 1] = cell_on_black[final_mask == 1]

    if save_intermediate:
        display_image("Segmented Nuclei (final)", masked_nuclei)
        save_image(masked_nuclei, "segmented_nuclei_final.png", output_dir)

    return final_mask, masked_nuclei

def extract_nuclei_only(
    img_color,
    output_dir=None,
    target_means=np.array([128, 128, 128]),
    target_stds=np.array([20, 20, 20]),
    min_size=30,
    save_intermediate=False,
    use_adaptive_thresholding=False  # New parameter
):
    """
    Extracts nuclei from the original image (img_color) and returns them on a black background.

    Parameters
    ----------
    img_color : np.ndarray
        Input color image (BGR), for example read by cv2.imread().
    output_dir : str or None
        Directory to save intermediate results. If None, no files are saved.
    target_means : np.ndarray
        Target mean values (LAB) for Reinhard color normalization (3-element array).
    target_stds : np.ndarray
        Target standard deviations (LAB) for Reinhard color normalization (3-element array).
    min_size : int
        Minimum object size (in pixels). Any object smaller than this will be removed.
    save_intermediate : bool
        If True, intermediate images are saved (and displayed) at each step.
    use_adaptive_thresholding : bool
        If True, uses adaptive thresholding instead of Otsu thresholding.

    Returns
    -------
    nuclei_only : np.ndarray
        The same shape as the input image, containing only the nuclei (unchanged) and
        the rest filled with black (0 pixel values).
    """
    output_dir = create_output_dir()

    # Step 0: Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(img_color, (5, 5), 0)
    if save_intermediate:
        save_image(blurred_image, "1_blurred.png", output_dir)
        display_image("Blurred Image", blurred_image)

    # 1. Perform color normalization (Reinhard). Assumes you have a function:
    #    reinhard_color_normalization(img, target_means, target_stds).
    normalized_image = reinhard_color_normalization(
        blurred_image, target_means, target_stds
    )
    if save_intermediate:
        save_image(normalized_image, "2_normalized.png", output_dir)
        display_image("Normalized Image", normalized_image)

    # 2. Extract the blue channel (OpenCV uses BGR format; index 0 is Blue).
    blue_channel = normalized_image[:, :, 0]
    if save_intermediate:
        save_image(blue_channel, "3_blue_channel.png", output_dir)
        display_image("Blue Channel", blue_channel, cmap="gray")

    # 3. Apply CLAHE (histogram equalization) with a limited clip limit for contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blue_channel_eq = clahe.apply(blue_channel.astype(np.uint8))
    if save_intermediate:
        save_image(blue_channel_eq, "4_blue_channel_eq.png", output_dir)
        display_image("CLAHE on Blue Channel", blue_channel_eq, cmap="gray")

    # 4. Thresholding (Otsu or Adaptive)
    if use_adaptive_thresholding:
        # Use adaptive thresholding
        nuclei_mask = cv2.adaptiveThreshold(
            blue_channel_eq,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Gaussian-weighted sum around neighborhood
            cv2.THRESH_BINARY,
            blockSize=11,  # Size of neighborhood (must be odd)
            C=2  # Constant subtracted from the mean
        )
        if save_intermediate:
            save_image(nuclei_mask, "5_adaptive_threshold.png", output_dir)
            display_image("Adaptive Thresholding", nuclei_mask, cmap="gray")
    else:
        # Use Otsu thresholding
        _, nuclei_mask = cv2.threshold(
            blue_channel_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if save_intermediate:
            save_image(nuclei_mask, "5_otsu_threshold.png", output_dir)
            display_image("Otsu Thresholding", nuclei_mask, cmap="gray")

    # 5. Perform morphological opening to remove small white specks.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_opened = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)
    if save_intermediate:
        save_image(mask_opened, "6_mask_opened.png", output_dir)
        display_image("Mask Opened", mask_opened, cmap="gray")

    # 6. (Optional) Dilate the mask. If a morphological_operations function is available, use that.
    mask_dil = cv2.dilate(mask_opened, kernel, iterations=2)
    if save_intermediate:
        save_image(mask_dil, "7_mask_dil.png", output_dir)
        display_image("Mask Dilated", mask_dil, cmap="gray")

    # 7. Remove small objects using skimage.remove_small_objects.
    mask_bool = mask_dil > 0
    mask_no_small = remove_small_objects(mask_bool, min_size=min_size)
    mask_no_small = mask_no_small.astype(np.uint8) * 255

    # 8. Fill holes using binary_fill_holes.
    mask_filled = binary_fill_holes(mask_no_small > 0).astype(np.uint8) * 255

    # 9. Overlay the mask on the original (normalized) image to extract only nuclei.
    mask_bool = mask_filled > 0
    nuclei_only = np.zeros_like(img_color)
    nuclei_only[mask_bool] = img_color[mask_bool]

    # 10. Optionally display and/or save the final results.
    if save_intermediate:
        save_image(mask_filled, "8_mask_filled.png", output_dir)
        display_image("Mask Final", mask_filled, cmap="gray")

        save_image(nuclei_only, "9_nuclei_only.png", output_dir)
        display_image("Nuclei Only", nuclei_only)

    # Return the final nuclei-only image on a black background.
    return nuclei_only

# def extract_nuclei_from_black_background(
#     img_color,
#     target_means=np.array([128, 128, 128]),
#     target_stds=np.array([20, 20, 20]),
#     min_size=50,
#     clip_limit=2.0,
#     percentile_threshold=10,
#     save_intermediate=False,
#     output_dir="output_images",
#     convert_white_bg_to_black=True,
#     threshold_for_white=250,
# ):
#     """
#     Extracts nuclei from an image with enhanced debugging to ensure proper segmentation.
#     """
#     # STEP 0: Optional conversion of white background to black
#     if convert_white_bg_to_black:
#         img_color = convert_white_background_to_black(img_color, threshold_for_white)
#         if save_intermediate:
#             save_image(img_color, "converted_white_to_black.png", output_dir)
#             display_image("Converted White to Black BG", img_color)

#     # Step 1: Apply Gaussian blur to reduce noise
#     blurred_image = cv2.GaussianBlur(img_color, (5, 5), 0)
#     if save_intermediate:
#         save_image(blurred_image, "blurred_image.png", output_dir)
#         display_image("Blurred Image", blurred_image)

#     # Step 2: Reinhard color normalization
#     normalized_image = reinhard_color_normalization(
#         blurred_image, target_means, target_stds
#     )
#     if save_intermediate:
#         save_image(normalized_image, "normalized_image.png", output_dir)
#         display_image("Normalized Image", normalized_image)

#     # Step 3: Extract the blue channel
#     blue_channel = normalized_image[:, :, 0]
#     if save_intermediate:
#         save_image(blue_channel, "blue_channel.png", output_dir)
#         display_image("Blue Channel", blue_channel, cmap="gray")

#     # Step 4: Apply CLAHE
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
#     blue_channel_eq = clahe.apply(blue_channel)
#     if save_intermediate:
#         save_image(blue_channel_eq, "blue_channel_eq.png", output_dir)
#         display_image("CLAHE on Blue Channel", blue_channel_eq, cmap="gray")

#     # Step 5: Determine the threshold based on a percentile
#     threshold_value = np.percentile(blue_channel_eq, percentile_threshold)
#     print(f"Threshold Value (Percentile {percentile_threshold}): {threshold_value}")

#     # Apply binary thresholding
#     _, nuclei_mask = cv2.threshold(
#         blue_channel_eq, threshold_value, 255, cv2.THRESH_BINARY
#     )
#     if save_intermediate:
#         save_image(nuclei_mask, "nuclei_mask.png", output_dir)
#         display_image("Nuclei Mask", nuclei_mask, cmap="gray")

#     # Step 6: Morphological opening to remove noise
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     mask_opened = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)
#     if save_intermediate:
#         save_image(mask_opened, "mask_opened.png", output_dir)
#         display_image("Morphologically Opened Mask", mask_opened, cmap="gray")

#     # Step 8: Remove small objects
#     mask_bool = mask_opened > 0
#     mask_no_small = remove_small_objects(mask_bool, min_size=min_size)
#     mask_no_small = mask_no_small.astype(np.uint8) * 255
#     if save_intermediate:
#         save_image(mask_no_small, "mask_no_small.png", output_dir)
#         display_image("Mask Without Small Objects", mask_no_small, cmap="gray")

#     # Step 7: Dilate the mask
#     mask_dil = cv2.dilate(mask_no_small, kernel, iterations=2)
#     if save_intermediate:
#         save_image(mask_dil, "mask_dil.png", output_dir)
#         display_image("Dilated Mask", mask_dil, cmap="gray")

#     # Step 9: Fill holes in the mask
#     mask_filled = binary_fill_holes(mask_no_small > 0).astype(np.uint8) * 255
#     if save_intermediate:
#         save_image(mask_filled, "mask_filled.png", output_dir)
#         display_image("Filled Mask", mask_filled, cmap="gray")

#     # Step 10: Apply the mask to the normalized image
#     mask_bool = mask_filled > 0
#     nuclei_only = np.zeros_like(img_color)
#     nuclei_only[mask_bool] = img_color[mask_bool]

#     if save_intermediate:
#         save_image(nuclei_only, "nuclei_only.png", output_dir)
#         display_image("Extracted Nuclei Only", nuclei_only)

#     return nuclei_only, mask_filled
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import show_anns
from sam import sam_model_registry, SamAutomaticMaskGenerator

def segment_nuclei_cellpose(
    img_array: np.ndarray,
    diameter: float = 200,
    model_type: str = "cyto3",
    gpu: bool = True,
    flow_threshold: float = 0.70,
    cellprob_threshold: float = 0.0,
    do_clahe: bool = True,
    show_plots: bool = True,
    channels: list = [0, 0],
    save_plots_path: str = None  # <-- new argument
):
    """
    Processes an image and segments cell nuclei using Cellpose.
    Optionally shows and/or saves the segmentation figure.

    Parameters
    ----------
    img_array : np.ndarray
        The input image of type np.uint8 (2D), typically grayscale
        (nuclei) extracted from a color image.
    diameter : float
        Estimated size of nuclei (in pixels). Used by the Cellpose model.
    model_type : str
        Model type for Cellpose ('cyto', 'cyto2', 'cyto3', 'nuclei', etc.).
    gpu : bool
        Whether to use the GPU (True) or CPU (False) for Cellpose.
    flow_threshold : float
        Cellpose flow threshold parameter.
    cellprob_threshold : float
        Cellpose cell probability threshold parameter.
    do_clahe : bool
        If True, applies CLAHE before segmentation.
    show_plots : bool
        If True, displays plots and figures using Matplotlib.
    channels : list
        The channels to pass to Cellpose. [0, 0] = single-channel grayscale.
    save_plots_path : str, optional
        If provided and show_plots=True, saves the segmentation figure to this path.

    Returns
    -------
    masks : np.ndarray
        A label array from the Cellpose segmentation (0=background, 1..N=distinct nuclei).
    flows, styles, diams_returned : additional Cellpose outputs.
    num_nuclei : int
        The number of detected nuclei.
    nuclei_areas : np.ndarray
        An array of nucleus areas (in pixels), one entry per nucleus.
    """
    import matplotlib.pyplot as plt
    from cellpose import models
    from skimage.color import label2rgb

    # Step 1: Prepare the image
    if img_array.dtype != np.uint8:
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)

    # If color, convert to grayscale
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_prepared = clahe.apply(img_array)
    else:
        img_prepared = img_array.copy()

    if show_plots:
        plt.figure()
        plt.imshow(img_prepared, cmap="gray")
        plt.title("CLAHE Image" if do_clahe else "Input Image")
        plt.axis("off")
        plt.show()

    # Step 2: Run Cellpose
    model = models.Cellpose(gpu=gpu, model_type=model_type)
    masks, flows, styles, diams_returned = model.eval(
        img_prepared,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    # Step 3: Visualize segmentation
    num_nuclei = np.max(masks)
    unique_labels, counts = np.unique(masks, return_counts=True)
    nuclei_areas = counts[1:]  # skip label 0 (background)

    if show_plots:
        colored_masks = label2rgb(masks, image=img_prepared, bg_label=0)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_prepared, cmap="gray")
        ax[0].set_title("Prepared Image")
        ax[0].axis("off")
        ax[1].imshow(colored_masks)
        ax[1].set_title("Cellpose Segmentation Masks")
        ax[1].axis("off")
        plt.tight_layout()

        # Optionally save
        if save_plots_path is not None:
            fig.savefig(save_plots_path)
            print(f"[Cellpose] Saved segmentation figure to: {save_plots_path}")

        plt.show()

        if num_nuclei > 0:
            plt.figure()
            plt.hist(nuclei_areas, bins=30)
            plt.xlabel("Nucleus area (pixels)")
            plt.ylabel("Frequency")
            plt.title(f"Nuclei area distribution (count: {num_nuclei})")
            plt.show()

    print(f"[Cellpose] Detected {num_nuclei} nuclei.")

    return masks, flows, styles, diams_returned, num_nuclei, nuclei_areas

# def sam_automatic_segmentation(
#     img_array,
#     sam_checkpoint,
#     model_type="vit_b",
#     device="cuda",
#     points_per_side=32,
#     pred_iou_thresh=0.9,
#     stability_score_thresh=0.96,
#     crop_n_layers=2,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,
#     show_plots=True,
# ):
#     """
#     Performs automatic image segmentation using the Segment Anything Model (SAM).

#     Parameters
#     ----------
#     img_array : np.ndarray
#         The input image, either BGR or RGB.
#         SAM internally prefers RGB format.
#     sam_checkpoint : str
#         Path to the .pth SAM checkpoint file.
#     model_type : str
#         Model type (e.g., 'vit_b', 'vit_h', 'vit_l').
#     device : str
#         Device to use: 'cuda' or 'cpu'.
#     points_per_side : int
#         The number of sample points per side for SamAutomaticMaskGenerator.
#     pred_iou_thresh : float
#         The prediction IoU threshold (confidence) for mask generation.
#     stability_score_thresh : float
#         The threshold for mask stability.
#     crop_n_layers : int
#         Number of layers to use when cropping the image into smaller patches.
#     crop_n_points_downscale_factor : int
#         Factor to downscale the number of sampling points when cropping.
#     min_mask_region_area : int
#         Minimum region area (in pixels) for a mask to be considered.
#     show_plots : bool
#         If True, displays images and masks using Matplotlib.

#     Returns
#     -------
#     masks : list
#         A list of masks generated by SamAutomaticMaskGenerator.
#         Each mask is a dictionary with keys such as 'segmentation', 'area', etc.
#     """
#     print("PyTorch version:", torch.__version__)
#     print("CUDA is available:", torch.cuda.is_available())

#     # Load the SAM model from the specified checkpoint
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)

#     # Convert the image to RGB if it appears to be in BGR
#     if len(img_array.shape) == 3 and img_array.shape[2] == 3:
#         # Heuristically assume the image is BGR if it comes from OpenCV
#         image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#     else:
#         # Assume it's already RGB or grayscale
#         image_rgb = img_array

#     # Optional display of the original (RGB) image
#     if show_plots:
#         plt.figure(figsize=(6, 6))
#         plt.imshow(image_rgb)
#         plt.title("Input Image (RGB)")
#         plt.axis("off")
#         plt.show()

#     # Configure the automatic mask generator
#     mask_generator = SamAutomaticMaskGenerator(
#         model=sam,
#         points_per_side=points_per_side,
#         pred_iou_thresh=pred_iou_thresh,
#         stability_score_thresh=stability_score_thresh,
#         crop_n_layers=crop_n_layers,
#         crop_n_points_downscale_factor=crop_n_points_downscale_factor,
#         min_mask_region_area=min_mask_region_area,
#     )

#     # Generate masks
#     print("Generating masks, please wait...")
#     masks = mask_generator.generate(image_rgb)
#     print(f"Generated {len(masks)} masks.")

#     # Optionally visualize results
#     if show_plots:
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image_rgb)
#         show_anns(masks)
#         plt.axis("off")
#         plt.title("SAM Automatic Masks")
#         plt.show()

#     return masks
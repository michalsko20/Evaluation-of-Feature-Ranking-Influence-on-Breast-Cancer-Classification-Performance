import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import mahotas

def extract_texture_features(
    image,
    mask,
    mode='all',                # Determines the output format: 'all' for lists, 'single' for scalars
    min_region_size=0,
    downsample_factor=1.0,
    compute_region_features=True,
    compute_global_glcm=True,
    compute_moments_hu=True,
    compute_zernike=True
):
    """
    Extracts texture, intensity, and shape features from the given image and mask.

    Parameters
    ----------
    image : np.ndarray
        The input image (grayscale or RGB) from which features are extracted.
    mask : np.ndarray
        The binary mask indicating regions of interest in the image.
    mode : str, optional
        Determines the output format:
        - 'all': Iterates over all regions and returns features as lists (one value per region).
        - 'single': Assumes the mask contains a single region (or considers the first region only) and
          returns features as scalars.
    min_region_size : int, optional
        Minimum region size to consider for feature extraction.
    downsample_factor : float, optional
        Factor by which to downsample the image and mask.
    compute_region_features : bool, optional
        Whether to compute region-based features.
    compute_global_glcm : bool, optional
        Whether to compute global GLCM (Gray Level Co-occurrence Matrix) features.
    compute_moments_hu : bool, optional
        Whether to compute Hu moments for each region.
    compute_zernike : bool, optional
        Whether to compute Zernike moments for each region.

    Returns
    -------
    features : dict
        A dictionary containing the extracted features. The structure depends on the mode parameter.
        - For mode='all': Keys are feature names, values are lists (one entry per region).
        - For mode='single': Keys are feature names, values are scalars.
        - Global GLCM features are always single values (floats).
    """
    # 1. Convert to grayscale if RGB
    if image.ndim == 3:
        gray_img = rgb2gray(image)
    else:
        gray_img = image.astype(float)

    # 2. Scale to [0..255]
    gray_img = (gray_img * 255).astype(np.uint8)

    # 2a. Downsampling if required
    if downsample_factor < 1.0:
        new_w = int(gray_img.shape[1] * downsample_factor)
        new_h = int(gray_img.shape[0] * downsample_factor)
        gray_img = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Resize the mask using nearest neighbor interpolation
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 3. Match the size of the mask to the image
    if mask.shape != gray_img.shape:
        mask = mask.astype(np.uint8)
        mask = cv2.resize(
            mask,
            (gray_img.shape[1], gray_img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Convert the mask to boolean
    mask = (mask > 0)

    # Prepare the results dictionary
    features = {}

    # ============================================
    # (A) Region-based features
    # ============================================
    if compute_region_features:
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask, intensity_image=gray_img)

        if mode == 'all':
            # Initialize lists for all region-based features
            features.update({
                "mean_intensity": [],
                "std_intensity": [],
                "skew_intensity": [],
                "kurtosis_intensity": [],
                "entropy_hist": [],
                "bounding_box_width": [],
                "bounding_box_height": [],
                "area": [],
                "perimeter": [],
                "eccentricity": [],
                "solidity": [],
                "extent": [],
                "major_axis_length": [],
                "minor_axis_length": [],
                "circularity": [],
                "zernike": []
            })

            for i in range(1, 8):  # Hu moments (7 values)
                features[f"hu_moment_{i}"] = []

            # Iterate over all regions
            for region in regions:
                if region.area < min_region_size:
                    continue

                region_intensity = region.intensity_image[region.image]
                mean_val = np.mean(region_intensity)
                std_val = np.std(region_intensity)
                skew_val = skew(region_intensity, axis=None)
                kurt_val = kurtosis(region_intensity, axis=None)

                hist, _ = np.histogram(region_intensity, bins=16, range=(0, 255))
                p = hist / (np.sum(hist) + 1e-10)
                entropy_val = -np.sum(p * np.log2(p + 1e-10))

                bbox = region.bbox
                bbox_width = bbox[3] - bbox[1]
                bbox_height = bbox[2] - bbox[0]

                area = region.area
                perimeter = region.perimeter
                eccentricity = region.eccentricity
                solidity = region.solidity
                extent = region.extent
                major_axis_length = region.major_axis_length
                minor_axis_length = region.minor_axis_length

                circularity_val = 4.0 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

                # Append to the feature lists
                features["mean_intensity"].append(mean_val)
                features["std_intensity"].append(std_val)
                features["skew_intensity"].append(skew_val)
                features["kurtosis_intensity"].append(kurt_val)
                features["entropy_hist"].append(entropy_val)
                features["bounding_box_width"].append(bbox_width)
                features["bounding_box_height"].append(bbox_height)
                features["area"].append(area)
                features["perimeter"].append(perimeter)
                features["eccentricity"].append(eccentricity)
                features["solidity"].append(solidity)
                features["extent"].append(extent)
                features["major_axis_length"].append(major_axis_length)
                features["minor_axis_length"].append(minor_axis_length)
                features["circularity"].append(circularity_val)

                if compute_moments_hu:
                    region_mask = (labeled_mask == region.label).astype(np.uint8)
                    try:
                        cv_mom = cv2.moments(region_mask, binaryImage=True)
                        hu_vals = cv2.HuMoments(cv_mom).flatten()
                        for i in range(7):
                            features[f"hu_moment_{i+1}"].append(hu_vals[i])
                    except:
                        for i in range(7):
                            features[f"hu_moment_{i+1}"].append(np.nan)

                if compute_zernike:
                    region_mask = (labeled_mask == region.label).astype(float)
                    try:
                        radius = min(bbox_width, bbox_height) / 2
                        zern = mahotas.features.zernike_moments(region_mask, radius=radius)
                        features["zernike"].append(zern)
                    except:
                        features["zernike"].append(np.nan)

        elif mode == 'single':
            if len(regions) == 0:
                return {}

            region = regions[0]
            if region.area < min_region_size:
                return {}

            region_intensity = region.intensity_image[region.image]
            mean_val = np.mean(region_intensity)
            std_val = np.std(region_intensity)
            skew_val = skew(region_intensity, axis=None)
            kurt_val = kurtosis(region_intensity, axis=None)

            hist, _ = np.histogram(region_intensity, bins=16, range=(0, 255))
            p = hist / (np.sum(hist) + 1e-10)
            entropy_val = -np.sum(p * np.log2(p + 1e-10))

            bbox = region.bbox
            bbox_width = bbox[3] - bbox[1]
            bbox_height = bbox[2] - bbox[0]

            area = region.area
            perimeter = region.perimeter
            eccentricity = region.eccentricity
            solidity = region.solidity
            extent = region.extent
            major_axis_length = region.major_axis_length
            minor_axis_length = region.minor_axis_length

            circularity_val = 4.0 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

            features.update({
                "mean_intensity": mean_val,
                "std_intensity": std_val,
                "skew_intensity": skew_val,
                "kurtosis_intensity": kurt_val,
                "entropy_hist": entropy_val,
                "bounding_box_width": bbox_width,
                "bounding_box_height": bbox_height,
                "area": area,
                "perimeter": perimeter,
                "eccentricity": eccentricity,
                "solidity": solidity,
                "extent": extent,
                "major_axis_length": major_axis_length,
                "minor_axis_length": minor_axis_length,
                "circularity": circularity_val
            })

            if compute_moments_hu:
                region_mask = (labeled_mask == region.label).astype(np.uint8)
                try:
                    cv_mom = cv2.moments(region_mask, binaryImage=True)
                    hu_vals = cv2.HuMoments(cv_mom).flatten()
                    for i in range(7):
                        features[f"hu_moment_{i+1}"] = hu_vals[i]
                except:
                    for i in range(7):
                        features[f"hu_moment_{i+1}"] = np.nan

            if compute_zernike:
                region_mask = (labeled_mask == region.label).astype(float)
                try:
                    radius = min(bbox_width, bbox_height) / 2
                    zern = mahotas.features.zernike_moments(region_mask, radius=radius)
                    for i, val in enumerate(zern):
                        features[f"zernike_{i}"] = val
                except:
                    features["zernike_0"] = np.nan

    # ============================================
    # (B) Global GLCM (single values for the entire mask)
    # ============================================
    if compute_global_glcm:
        try:
            gray_masked = gray_img.copy()
            gray_masked[~mask] = 0

            glcm = graycomatrix(
                gray_masked,
                distances=[1, 2, 3],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True
            )
            features["GLCM_contrast"] = float(np.mean(graycoprops(glcm, 'contrast')))
            features["GLCM_homogeneity"] = float(np.mean(graycoprops(glcm, 'homogeneity')))
            features["GLCM_energy"] = float(np.mean(graycoprops(glcm, 'energy')))
            features["GLCM_correlation"] = float(np.mean(graycoprops(glcm, 'correlation')))
        except:
            features.update({
                "GLCM_contrast": np.nan,
                "GLCM_homogeneity": np.nan,
                "GLCM_energy": np.nan,
                "GLCM_correlation": np.nan
            })

    return features

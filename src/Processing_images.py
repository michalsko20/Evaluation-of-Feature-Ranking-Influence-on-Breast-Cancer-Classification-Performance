import os
from scipy.stats import variation  # CV = coefficient of variation
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label


from utils import display_image
from Segmentation_functions_classic import process_and_visualize_he_image_blue_channel_with_morphology, extract_nuclei_only
from Segmentations_models import segment_nuclei_cellpose
from feature_extraction import extract_texture_features

# Import or define all custom functions used:
# - process_and_visualize_he_image_blue_channel_with_morphology
# - extract_nuclei_only
# - extract_nuclei_from_black_background
# - fuzzy_cmeans_in_rgb_2clusters
# - segment_nuclei_cellpose
# - extract_texture_features


def process_single_image(image_path, output_dir):
    """
    Processes a single image, performs segmentation, extracts features,
    computes diversity metrics, and saves the results in a dedicated folder.

    Parameters
    ----------
    image_path : str
        Path to the image to process.
    output_dir : str
        Path to the directory where the results will be saved.

    """
    try:
        # Create a dedicated folder for results
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # 1. Load and process the image (e.g., with blue channel and morphology)
        nuclei_on_white = process_and_visualize_he_image_blue_channel_with_morphology(
            image_path=image_path,
            save_intermediate=False,
            desired_size=None,
            apply_median_filter=True,
            median_kernel_size=3,
            increase_contrast=False,
            alpha=1.3,
            beta=15,
            adjust_blue_after_norm=True,
            blue_factor=1,
            blue_offset=0,
            morph_kernel_size=3
        )
        only_nucleis_1 = extract_nuclei_only(nuclei_on_white)
        display_image('only', only_nucleis_1)

        # Save the pre-segmented image
        pre_segmentation_image_path = os.path.join(image_output_dir, f"{image_name}_pre_segmented.png")
        cv2.imwrite(pre_segmentation_image_path, cv2.cvtColor(only_nucleis_1, cv2.COLOR_BGR2RGB))

        # 2. Segmentation with Cellpose
        seg_figure_path = os.path.join(image_output_dir, f"{image_name}_cellpose_masks.png")
        masks, flows, styles, diams_returned, num_nuclei, nuclei_areas = segment_nuclei_cellpose(
            only_nucleis_1,
            show_plots=True,
            diameter=80,
            save_plots_path=seg_figure_path
        )

        # Adjust mask dimensions to match the original image if necessary
        if masks.shape[:2] != only_nucleis_1.shape[:2]:
            masks = cv2.resize(
                masks,
                (only_nucleis_1.shape[1], only_nucleis_1.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # 3. Extract features for each nucleus ("single" mode)
        labeled_mask = label(masks)
        texture_features = []

        for region_id in range(1, np.max(labeled_mask) + 1):
            region_mask = (labeled_mask == region_id)
            features = extract_texture_features(
                image=only_nucleis_1,
                mask=region_mask,
                mode='single',
                downsample_factor=0.5
            )
            features["region_id"] = region_id
            texture_features.append(features)

        individual_features_df = pd.DataFrame(texture_features)

        # 4. Diversity metrics (CV, std, min, max)
        diversity_features = {}
        numeric_cols = individual_features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_series = individual_features_df[col].dropna()
            diversity_features[f"cv_{col}"] = variation(col_series) if len(col_series) > 1 else np.nan
            diversity_features[f"std_{col}"] = col_series.std()
            diversity_features[f"min_{col}"] = col_series.min()
            diversity_features[f"max_{col}"] = col_series.max()

        # 5. Hierarchical analysis of nuclei sizes (small, medium, large)
        if "area" not in numeric_cols:
            raise ValueError("'area' column is missing or not numeric in the extracted features.")

        small_nuclei = individual_features_df[individual_features_df["area"] <= individual_features_df["area"].quantile(0.33)]
        medium_nuclei = individual_features_df[
            (individual_features_df["area"] > individual_features_df["area"].quantile(0.33)) &
            (individual_features_df["area"] <= individual_features_df["area"].quantile(0.66))
        ]
        large_nuclei  = individual_features_df[individual_features_df["area"] > individual_features_df["area"].quantile(0.66)]

        group_features = {
            "small_nuclei_mean":  small_nuclei.mean(numeric_only=True).to_dict(),
            "medium_nuclei_mean": medium_nuclei.mean(numeric_only=True).to_dict(),
            "large_nuclei_mean":  large_nuclei.mean(numeric_only=True).to_dict()
        }

        # 6. Global features for the entire image
        full_image_mask = np.ones(only_nucleis_1.shape[:2], dtype=bool)
        full_image_features = extract_texture_features(
            image=only_nucleis_1,
            mask=full_image_mask,
            mode='single',
            downsample_factor=0.5
        )

        # 7. Finalize the feature vector
        final_feature_vector = pd.concat([
            pd.Series(full_image_features),
            pd.Series(individual_features_df.mean(numeric_only=True).to_dict()),
            pd.Series(diversity_features),
            pd.Series(group_features["small_nuclei_mean"]),
            pd.Series(group_features["medium_nuclei_mean"]),
            pd.Series(group_features["large_nuclei_mean"])
        ]).to_dict()

        print("[DEBUG] final_feature_vector:")
        for k, v in final_feature_vector.items():
            if isinstance(v, list):
                print(f"   {k}: list of length {len(v)}")
            else:
                print(f"   {k}: {v} (type: {type(v)})")

        # Save the feature vector as a DataFrame
        final_feature_df = pd.DataFrame(final_feature_vector, index=[0])

        # 8. Save results to CSV
        csv_name = f"{image_name}_features.csv"
        csv_path = os.path.join(image_output_dir, csv_name)
        final_feature_df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"[ERROR] Error processing {image_path}: {e}")
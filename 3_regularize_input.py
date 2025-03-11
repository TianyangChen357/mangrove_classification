import os
import numpy as np
import rasterio
from rasterio.windows import Window
import sys

"""
Script Name: 3_regularize_input.py
Description: 
This script splits merged Landsat GeoTIFFs and corresponding mask files into 256x256 pixel subsets. 
It **only saves subsets** where:
    1. No NoData values are present in the imagery.
    2. At least one mangrove pixel is present in the mask (sum(mask) â‰  0).

Usage:
    python 3_regularize_input.py scene_list_display_id.txt

Requirements:
- Merged Landsat images should be stored in `1_imagery_merge/`
- Corresponding masks should be in `2_Mask/`
- A text file (`scene_list_display_id.txt`) containing Landsat scene names (one per line)

Example:
    1. Prepare `scene_list_display_id.txt` with the following content:
        LT05_L2SP_008046_20100122_20200825_02_T1
        LT05_L2SP_009047_20100215_20200825_02_T1
        LT05_L2SP_010048_20100310_20200825_02_T1

    2. Run the script:
        python 3_regularize_input.py scene_list_display_id.txt

Output:
- **Only valid subsets** will be saved in `3_imagery_subsets/` and `3_mask_subsets/`
- **Invalid subsets (with NoData or no mangrove pixels) will be skipped**

Notes:
- If an input file is missing, the script will skip processing that scene.
- The script ensures that all subsets are 256x256 with necessary padding.
"""

def create_valid_subsets(scene_name, imagery_file, mask_file, imagery_output_dir, mask_output_dir, subset_size=256):
    """
    Splits a merged Landsat GeoTIFF and mask into subsets, **only saving valid subsets** 
    (no NoData values and containing mangroves).

    Parameters:
        scene_name (str): Name of the Landsat scene.
        imagery_file (str): Path to the merged imagery GeoTIFF file.
        mask_file (str): Path to the mask GeoTIFF file.
        imagery_output_dir (str): Directory to save valid imagery subsets.
        mask_output_dir (str): Directory to save valid mask subsets.
        subset_size (int): Size of each subset (default: 256).
    """
    os.makedirs(imagery_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    with rasterio.open(imagery_file) as src_imagery, rasterio.open(mask_file) as src_mask:
        width, height = src_imagery.width, src_imagery.height
        transform, crs, dtype = src_imagery.transform, src_imagery.crs, src_imagery.dtypes[0]
        count, nodata = src_imagery.count, src_imagery.nodata

        num_rows = (height + subset_size - 1) // subset_size
        num_cols = (width + subset_size - 1) // subset_size

        for row in range(num_rows):
            for col in range(num_cols):
                row_start, col_start = row * subset_size, col * subset_size
                row_end, col_end = min((row + 1) * subset_size, height), min((col + 1) * subset_size, width)

                window_height, window_width = row_end - row_start, col_end - col_start
                window = Window(col_start, row_start, window_width, window_height)

                imagery_data = src_imagery.read(window=window, boundless=True, fill_value=0)
                mask_data = src_mask.read(1, window=window, boundless=True, fill_value=0)

                # Skip subsets with NoData values
                if nodata is not None and np.any(imagery_data == nodata):
                    print(f"âŒ Skipping {scene_name}_{row}_{col}: Contains NoData values.")
                    continue

                # Skip subsets without mangrove presence
                if np.sum(mask_data) == 0:
                    print(f"âŒ Skipping {scene_name}_{row}_{col}: No mangrove pixels.")
                    continue

                # Apply padding
                padded_imagery = np.zeros((count, subset_size, subset_size), dtype=dtype)
                padded_imagery[:, :window_height, :window_width] = imagery_data

                padded_mask = np.zeros((subset_size, subset_size), dtype=np.uint8)
                padded_mask[:window_height, :window_width] = mask_data

                # Save valid imagery subset
                imagery_output_file = os.path.join(imagery_output_dir, f"{scene_name}_{row}_{col}.tif")
                with rasterio.open(
                    imagery_output_file, "w",
                    driver="GTiff",
                    height=subset_size, width=subset_size,
                    count=count, dtype=dtype,
                    crs=crs, transform=src_imagery.window_transform(window),
                    nodata=nodata
                ) as dst:
                    dst.write(padded_imagery)

                # Save valid mask subset
                mask_output_file = os.path.join(mask_output_dir, f"{scene_name}_{row}_{col}.tif")
                with rasterio.open(
                    mask_output_file, "w",
                    driver="GTiff",
                    height=subset_size, width=subset_size,
                    count=1, dtype=np.uint8,
                    crs=crs, transform=src_mask.window_transform(window),
                    nodata=0
                ) as dst:
                    dst.write(padded_mask, 1)

                # print(f"âœ… Saved valid subset: {imagery_output_file}, {mask_output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 3_regularize_input.py <scene_list_file>")
        sys.exit(1)

    scene_list_file = sys.argv[1]

    with open(scene_list_file, "r") as f:
        scene_names = [line.strip() for line in f.readlines() if line.strip()]

    imagery_input_dir = "./1_imagery_merge"
    mask_input_dir = "./2_Mask"
    imagery_output_dir = "./3_imagery_subsets"
    mask_output_dir = "./3_mask_subsets"

    for scene_name in scene_names:
        imagery_file = os.path.join(imagery_input_dir, f"{scene_name}_merged.tif")
        mask_file = os.path.join(mask_input_dir, f"{scene_name}_mask.TIF")

        if os.path.exists(imagery_file) and os.path.exists(mask_file):
            # print(f"ðŸ›  Processing: {scene_name}...")
            create_valid_subsets(scene_name, imagery_file, mask_file, imagery_output_dir, mask_output_dir)
        else:
            print(f"âš ï¸ Skipping {scene_name}: Missing imagery or mask file.")

    print("âœ… All processing completed!")

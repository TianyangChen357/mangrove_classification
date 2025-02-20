import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

def create_subsets_from_merged(scene_name, input_file, output_dir, subset_size=256):
    """
    Splits a merged Landsat GeoTIFF into 256x256 subsets, with padding if needed, and saves each subset as a separate GeoTIFF.

    Parameters:
        scene_name (str): Name of the Landsat scene (prefix).
        input_file (str): Path to the merged GeoTIFF file.
        output_dir (str): Directory where the subsets will be saved.
        subset_size (int): Size of each subset (default: 256).
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        dtype = src.dtypes[0]
        count = src.count
        nodata = src.nodata
        # Calculate number of rows and columns of subsets
        num_rows = (height + subset_size - 1) // subset_size
        num_cols = (width + subset_size - 1) // subset_size

        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate the window bounds
                row_start = row * subset_size
                col_start = col * subset_size

                row_end = min((row + 1) * subset_size, height)
                col_end = min((col + 1) * subset_size, width)

                window_height = row_end - row_start
                window_width = col_end - col_start

                # Read data from the window
                window = Window(col_start, row_start, window_width, window_height)
                data = src.read(window=window, boundless=True, fill_value=0)

                # Adjust transform for the subset
                subset_transform = src.window_transform(window)

                # Apply padding if necessary
                padded_data = np.zeros((count, subset_size, subset_size), dtype=dtype)
                padded_data[:, :window_height, :window_width] = data

                # Create the output file name
                output_file = os.path.join(output_dir, f"{scene_name}_{row}_{col}.tif")

                # Save the subset as a GeoTIFF
                with rasterio.open(
                    output_file,
                    "w",
                    driver="GTiff",
                    height=subset_size,
                    width=subset_size,
                    count=count,
                    dtype=dtype,
                    crs=crs,
                    transform=subset_transform,
                    nodata=nodata
                ) as dst:
                    dst.write(padded_data)

                print(f"Saved subset: {output_file}")

# Example usage
if __name__ == "__main__":
# Example usage
    scene_list_file = "scene_list_display_id_test4.txt"
    # Load scene names from the file
    with open(scene_list_file, "r") as f:
        scene_names = [line.strip() for line in f.readlines() if line.strip()]
# imagery subset
    input_directory = "./1_imagery_merge"  # Directory containing Landsat GeoTIFF files
    output_directory = "./3_imagery_subsets"  # Directory to save the subsets
    os.makedirs(output_directory, exist_ok=True)
    for scene_name in scene_names:
        input_file=os.path.join(input_directory,f'{scene_name}_merged.tif')
        create_subsets_from_merged(scene_name, input_file, output_directory)
# mask subset 
    input_directory = "./2_Mask"  # Directory containing Landsat GeoTIFF files
    output_directory = "./3_mask_subsets"  # Directory to save the subsets
    os.makedirs(output_directory, exist_ok=True)
    for scene_name in scene_names:
        input_file=os.path.join(input_directory,f'{scene_name}_mask.TIF')
        create_subsets_from_merged(scene_name, input_file, output_directory)
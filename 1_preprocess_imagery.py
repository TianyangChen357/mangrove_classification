import os
import numpy as np
import rasterio
import tarfile

def extract_tar_files(scene_name, tar_dir, extract_dir):
    """
    Extracts the contents of a TAR file corresponding to a Landsat scene.

    Parameters:
        scene_name (str): Name of the Landsat scene (prefix).
        tar_dir (str): Directory containing TAR files.
        extract_dir (str): Directory where extracted files will be stored.
    """
    tar_file_path = os.path.join(tar_dir, f"{scene_name}.tar")
    scene_extract_path = os.path.join(extract_dir, scene_name)

    if not os.path.exists(scene_extract_path):
        os.makedirs(scene_extract_path, exist_ok=True)

    if os.path.exists(tar_file_path):
        print(f"Extracting {tar_file_path} to {scene_extract_path}...")
        with tarfile.open(tar_file_path, "r") as tar:
            tar.extractall(path=scene_extract_path)
        print(f"Extraction complete: {scene_extract_path}")
    else:
        print(f"Warning: TAR file not found for {scene_name} in {tar_dir}")

def merge_scene_bands_to_geotiff(scene_name, input_dir, output_dir):
    """
    Merges Landsat bands for a single scene and saves the merged array as a multi-band GeoTIFF file.

    Parameters:
        scene_name (str): Name of the Landsat scene (prefix).
        input_dir (str): Directory containing Landsat GeoTIFF files.
        output_dir (str): Directory where the merged GeoTIFF file will be saved.

    Returns:
        str: Path to the saved GeoTIFF file.
    """
    band_indices = [1, 2, 3, 4, 5, 7]  # Channels to include
    merged_data = []
    transform = None
    crs = None

    scene_path = os.path.join(input_dir, scene_name)
    if not os.path.exists(scene_path):
        print(f"Warning: Extracted scene directory not found: {scene_path}")
        return None

    for band_idx in band_indices:
        file_name = f"{scene_name}_SR_B{band_idx}.TIF"
        file_path = os.path.join(scene_path, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: Band file not found: {file_path}")
            continue

        with rasterio.open(file_path) as src:
            band_data = src.read(1)
            band_data[band_data == src.nodata] = 0
            merged_data.append(band_data)
            if transform is None:
                transform = src.transform
                crs = src.crs

    if not merged_data:
        print(f"Skipping {scene_name} due to missing bands.")
        return None

    merged_array = np.stack(merged_data, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}_merged.tif")

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=merged_array.shape[1],
        width=merged_array.shape[2],
        count=merged_array.shape[0],
        dtype=merged_array.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        for i in range(merged_array.shape[0]):
            dst.write(merged_array[i, :, :], i + 1)

    print(f"Merged GeoTIFF for {scene_name} saved to {output_path}")
    return output_path


def process_all_scenes_to_geotiff(scene_names, tar_dir, extract_dir, output_dir):
    """
    Extracts TAR files, processes Landsat scenes, merges bands, and saves GeoTIFFs.

    Parameters:
        scene_names (list of str): List of Landsat scene names.
        tar_dir (str): Directory containing TAR files.
        extract_dir (str): Directory to extract TAR files.
        output_dir (str): Directory to save merged GeoTIFFs.
    """
    for scene_name in scene_names:
        try:
            extract_tar_files(scene_name, tar_dir, extract_dir)
            merge_scene_bands_to_geotiff(scene_name, extract_dir, output_dir)
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")

if __name__ == "__main__":
    # Read scene names from file
    scene_list_file = "scene_list_display_id_test4.txt"
    if os.path.exists(scene_list_file):
        with open(scene_list_file, "r") as file:
            scene_names = [line.strip() for line in file if line.strip()]
    else:
        print(f"Error: {scene_list_file} not found.")
        scene_names = []

    tar_directory = "./0_landsat_downloads"  # TAR file location
    extract_directory = "./1_imagery"  # Directory to extract files
    output_directory = "./1_imagery_merge"  # Output directory for merged TIFFs
    os.makedirs(extract_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    process_all_scenes_to_geotiff(scene_names, tar_directory, extract_directory, output_directory)

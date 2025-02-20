import os
import rasterio
from rasterio.merge import merge
import random

def merge_predictions_by_scene(input_dir, output_dir, test_scenes):
    """
    Merges predicted masks for each scene into a single raster.
    
    Parameters:
        input_dir (str): Directory containing individual predicted mask tiles.
        output_dir (str): Directory to save the merged masks per scene.
        test_scenes (list): List of scene names to merge.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for scene in test_scenes:
        print(f'Merging scene {scene}')
        
        file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith(scene) and f.endswith(".tif")]
        
        if not file_list:
            print(f"No files found for scene {scene}, skipping...")
            continue

        # Open files using context manager to avoid exceeding file limit
        src_files = []
        try:
            for fp in file_list:
                src = rasterio.open(fp)
                src_files.append(src)

            mosaic, out_trans = merge(src_files)

            with rasterio.open(file_list[0]) as src:
                out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": 1
            })

            output_path = os.path.join(output_dir, f"{scene}_merged.tif")
            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(mosaic[0], 1)  # Ensure the correct number of dimensions is used

            print(f"Merged scene {scene} saved at {output_path}")

        finally:
            # Ensure all files are closed
            for src in src_files:
                src.close()

if __name__ == "__main__":
    input_directory = "./5_test_predictions"  # Directory containing predicted mask subsets
    output_directory = "./6_merged_predictions"  # Directory to store merged outputs
    
    scene_list_file = "scene_list_display_id_test4.txt"
    with open(scene_list_file, "r") as f:
        scene_names = [line.strip() for line in f.readlines() if line.strip()]
    
    split_ratio = 0.8  # Adjust as needed
    SEED = 42
    random.seed(SEED)
    # Shuffle the scenes to ensure randomness (but with a fixed seed for reproducibility)
    random.shuffle(scene_names)

    # Split the data
    split_index = int(len(scene_names) * split_ratio)
    test_scenes = scene_names[split_index:]
    # test_scenes = [
    #     'LT05_L2SP_107070_20080428_20200829_02_T1', 
    #     'LT05_L2SP_121058_20091010_20200825_02_T2', 
    #     'LT05_L2SP_035043_20080926_20200829_02_T1', 
    #     'LT05_L2SP_160042_20080517_20200829_02_T1',
    # ]  # Update with actual test scenes
    
    merge_predictions_by_scene(input_directory, output_directory, test_scenes)

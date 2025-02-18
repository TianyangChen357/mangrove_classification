import os
import rasterio
from rasterio.merge import merge

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
        print(f'merging scene {scene}')
        file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith(scene) and f.endswith(".tif")]
        
        if not file_list:
            print(f"No files found for scene {scene}, skipping...")
            continue
        
        src_files = [rasterio.open(fp) for fp in file_list]
        mosaic, out_trans = merge(src_files)
        
        with rasterio.open(file_list[0]) as src:
            out_meta = src.meta.copy()
        
        out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans, "count": 1})
        
        output_path = os.path.join(output_dir, f"{scene}_merged.tif")
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic[0], 1)  # Ensure the correct number of dimensions is used
        
        print(f"Merged scene {scene} saved at {output_path}")

if __name__ == "__main__":
    input_directory = "./test_predictions"  # Directory containing predicted mask subsets
    output_directory = "./merged_predictions"  # Directory to store merged outputs
    
    test_scenes = [
        "LT05_L2SP_008046_20100122_20200825_02_T1_SR",
        "LT05_L2SP_009046_20070918_20200829_02_T1_SR"
    ]  # Update with actual test scenes
    
    merge_predictions_by_scene(input_directory, output_directory, test_scenes)

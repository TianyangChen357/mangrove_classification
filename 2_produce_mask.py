import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import numpy as np
import os
from rasterio.warp import transform_bounds

def create_mask_for_landsat(landsat_path, shapefile_path, output_mask_path):
    """Creates a binary mask while preserving NoData pixels in the Landsat raster."""
    
    # Step 1: Load the merged Landsat scene
    with rasterio.open(landsat_path) as src:
        raster_crs = src.crs  # Get CRS of the raster
        raster_bounds = src.bounds  # Get bounding box of the raster
        raster_transform = src.transform  # Get raster transform
        raster_shape = (src.height, src.width)  # Get raster dimensions
        nodata_value = src.nodata  # Get NoData value
        landsat_data = src.read(1)  # Read the first band (assumed representative)

    # Step 2: Load the mangrove shapefile
    mangrove_gdf = gpd.read_file(shapefile_path)
###########################################
    mangrove_crs = mangrove_gdf.crs

    # Step 4: Transform the raster bounding box to the GMW coordinate system
    if raster_crs != mangrove_crs:
        raster_bounds_transformed = transform_bounds(raster_crs, mangrove_crs,
                                                    raster_bounds.left, raster_bounds.bottom,
                                                    raster_bounds.right, raster_bounds.top)
    else:
        raster_bounds_transformed = raster_bounds  # No transformation needed

    # Step 5: Create a bounding box in the GMW CRS
    raster_bbox_in_gmw_crs = box(*raster_bounds_transformed)

    # Step 6: Clip the mangrove shapefile using the transformed raster bounding box
    mangrove_clipped = mangrove_gdf[mangrove_gdf.intersects(raster_bbox_in_gmw_crs)]
    mangrove_clipped = mangrove_clipped.to_crs(raster_crs)

  ##########################################
    # Step 5: Rasterize the mangrove polygons into a mask
    mangrove_shapes = [(geom, 1) for geom in mangrove_clipped.geometry]
    print(len(mangrove_shapes))
    if len(mangrove_shapes)!=0:
        mask = rasterize(
            shapes=mangrove_shapes,
            out_shape=raster_shape,
            transform=raster_transform,
            fill=0,  # Non-mangrove pixels default to 0
            dtype=np.uint8
    )

        # Step 6: Preserve NoData pixels from the Landsat scene
        if nodata_value is not None:
            mask[landsat_data == nodata_value] = 255  # Assign NoData pixels a distinct value (255)

        # Step 7: Save the mask as a GeoTIFF with NoData value set to 255
        with rasterio.open(
            output_mask_path,
            'w',
            driver='GTiff',
            height=raster_shape[0],
            width=raster_shape[1],
            count=1,
            dtype=np.uint8,
            crs=raster_crs,
            transform=raster_transform,
            nodata=255  # Set NoData value in the mask
        ) as dst:
            dst.write(mask, 1)

        print(f"✅ Mask saved: {output_mask_path} (NoData preserved)")

if __name__ == "__main__":
    # Read scene names from file
    scene_list_file = "scene_list_display_id_test4.txt"

    # Define directories
    imagery_merge_dir = "./1_imagery_merge/"  # Folder containing merged Landsat images
    mask_dir = "./2_Mask/"  # Output folder for masks
    mangrove_dir = "./0_GMW/"  # Folder containing mangrove shapefiles by year

    # Load scene names from the file
    with open(scene_list_file, "r") as f:
        scene_names = [line.strip() for line in f.readlines() if line.strip()]

    for scene_name in scene_names:
        # Extract the year from the scene name (e.g., "LT05_L2SP_008046_20100122_20200825_02_T1")
        year = scene_name.split("_")[3][:4]

        # Construct file paths
        landsat_scene_path = os.path.join(imagery_merge_dir, f"{scene_name}_merged.tif")  # Use merged image
        mangrove_shapefile_path = os.path.join(mangrove_dir, f"GMW_v3_{year}/gmw_v3_{year}_vec.shp")
        output_mask_path = os.path.join(mask_dir, f"{scene_name}_mask.TIF")

        # Check if required files exist
        if os.path.exists(landsat_scene_path) and os.path.exists(mangrove_shapefile_path):
            print(f"Processing: {scene_name} with {mangrove_shapefile_path}...")
            create_mask_for_landsat(landsat_scene_path, mangrove_shapefile_path, output_mask_path)
        else:
            print(f"⚠️ Skipping {scene_name}: Missing merged imagery or mangrove shapefile")

    print("✅ All processing completed!")
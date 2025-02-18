from landsatxplore.api import API
import geopandas as gpd
from tqdm import tqdm

# Load your USGS Earth Explorer credentials
username = "tchen19"
password = "Cty15830061892"

# Initialize API
api = API(username, password)

# Load Global Mangrove Watch (GMW) shapefile
gmw = gpd.read_file("./0_GMW/GMW_v3_2010/gmw_v3_2010_vec.shp")
gmw = gmw.sample(n=50)
# print(gmw)
# Query parameters
start_date = "2007-01-01"
end_date = "2020-12-31"
dataset = "LANDSAT_TM_C2_L2"  # Landsat Thematic Mapper C2 Level-2

# Store unique scene IDs in a set
scene_ids = set()
display_ids= set()
# Loop through each polygon in GMW with tqdm progress bar
tqdm_bar = tqdm(gmw.iterrows(), total=len(gmw), desc="Processing polygons")

for index, row in tqdm_bar:
    bbox = row.geometry.bounds  # Get bounding box (minx, miny, maxx, maxy)
    
    # Query Landsat images for this region with 0% cloud cover
    scenes = api.search(
        dataset=dataset,
        bbox=[bbox[0], bbox[1], bbox[2], bbox[3]],  # (lat_min, lon_min, lat_max, lon_max)
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=0  # Set cloud cover to 0% but it seems this one does not work properly
    )
    
    # Update tqdm progress bar with the number of scenes found
    tqdm_bar.set_description(f"Scenes found: {len(scene_ids)}")

    # Add unique scene IDs to the set
    scene_ids.update(scene["entity_id"] for scene in scenes if scene["land_cloud_cover"] == 0)
    display_ids.update(scene["display_id"] for scene in scenes if scene["land_cloud_cover"] == 0)


# Save to a text file
with open("scene_list_entity_id.txt", "w") as f:
    for scene in scene_ids:
        f.write(scene + "\n")
with open("scene_list_display_id.txt", "w") as f:
    for scene in display_ids:
        f.write(scene + "\n")

# Logout from API
api.logout()

print(f"Found {len(scene_ids)} unique cloud-free Landsat TM C2L2 scenes.")

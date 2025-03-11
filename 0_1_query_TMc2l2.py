import requests
import geopandas as gpd
import sys
import random
import json

"""
Script Name: 0_1_query_TMc2l2.py
Description:
This script queries Landsat scenes from USGS M2M API and selects only **Tier 1 (T1)** scenes.
It automatically **excludes Tier 2 (T2) scenes** from the selection.

Example

python 0_1_query_TMc2l2.py 500 landsat_tm_c2_l2
Output:
- **Valid `displayId` and `entityId` scenes** are saved as a list in `scene_list_display_id.txt` and `scene_list_entity_id.txt`
"""

# USGS M2M API Endpoint
SERVICE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"

# ?? Hardcoded USGS Credentials
USERNAME = "tchen19"
TOKEN = "L7n6RhXN2gvJUsK4_z@V9bneeSZ7FuZEppG06gAm7e7sDfRpDJXWAf@y@uGzxpjG"

# Function to send API requests
def send_request(url, data, api_key=None):
    headers = {"X-Auth-Token": api_key} if api_key else {}
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        if response_data["errorCode"] is None:
            return response_data["data"]
        else:
            print(f"? API Error: {response_data['errorMessage']}")
            return None
    else:
        print(f"? HTTP Error {response.status_code} for {url}")
        return None

# Step 1: Authenticate and get API Key
# print("\n?? Logging in to USGS M2M API...")
api_key = send_request(SERVICE_URL + "login-token", {"username": USERNAME, "token": TOKEN})
if not api_key:
    print("? Authentication failed. Exiting...")
    exit()

# print(f"? API Key: {api_key}\n")

# Step 2: Load Global Mangrove Watch (GMW) shapefile
gmw = gpd.read_file("./0_GMW/GMW_v3_2010/gmw_v3_2010_vec.shp")

# User-specified number of required scenes
required_scenes = int(sys.argv[1])
dataset = sys.argv[2]

# Store selected scenes
location_scenes = {}
scene_list = []

# Step 3: Keep sampling locations until we reach the required number of scenes
while len(location_scenes) < required_scenes:
    # Randomly sample a location from the dataset
    sampled_location = gmw.sample(n=1).iloc[0]
    index = sampled_location.name
    bbox = sampled_location.geometry.bounds  # Get bounding box (minx, miny, maxx, maxy)
    # print(f"?? Processing Bounding Box: {bbox}")

    # ? Corrected Payload
    payload = {
        "datasetName": dataset,
        "maxResults": 5,  # Increase to find more valid results
        "startingNumber": 1,
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"longitude": bbox[0], "latitude": bbox[1]},
                "upperRight": {"longitude": bbox[2], "latitude": bbox[3]}
            },
            "cloudCoverFilter": {
                "min": 0,
                "max": 0,  # Allow slight cloud cover if needed
                "includeUnknown": True
            },
            "acquisitionFilter": {
                "start": "2007-01-01",
                "end": "2010-12-31"
            }
        },
        "metadataType": "summary",
        "sortDirection": "ASC"
    }

    # Make the request
    scenes = send_request(SERVICE_URL + "scene-search", payload, api_key)

    # Process valid cloud-free Tier 1 (T1) scenes
    if scenes and "results" in scenes:
        for scene in scenes["results"]:
            if scene["displayId"].endswith("_T1"):  # ? Select only Tier 1 scenes
                location_scenes[index] = {
                    "displayId": scene["displayId"],
                    "entityId": scene["entityId"]
                }
                scene_list.append((scene["displayId"], scene["entityId"]))
                # print(f"? Selected: {scene['displayId']} (T1) with Entity ID: {scene['entityId']}")
                break  # ?? Stop after selecting first valid scene

    # print(f"?? Progress: {len(location_scenes)} out of {required_scenes} scenes selected.")

# Step 4: Save `displayId` and `entityId` to text files
with open("scene_list_display_id.txt", "w") as f:
    for display_id, _ in scene_list:
        f.write(display_id + "\n")

with open("scene_list_entity_id.txt", "w") as f:
    for _, entity_id in scene_list:
        f.write(entity_id + "\n")

# Step 5: Logout from API
send_request(SERVICE_URL + "logout", {}, api_key)
# print("? Successfully logged out.")
print(f"? Successfully selected {len(location_scenes)} unique cloud-free Landsat TM C2L2 Tier 1 (T1) scenes.")

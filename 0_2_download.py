from landsatxplore.earthexplorer import EarthExplorer

# Your USGS Earth Explorer credentials
username = "tchen19"
password = "Cty15830061892"

# Input file with Display IDs
input_file = "scene_list_display_id_test4.txt"

# Output directory for downloads
output_dir = "0_landsat_downloads"

# Read Display IDs from file
with open(input_file, "r") as f:
    display_ids = [line.strip() for line in f.readlines() if line.strip()]

# Initialize EarthExplorer for downloading
ee = EarthExplorer(username, password)

# Download each scene by Display ID
for display_id in display_ids:
    print(f"Downloading {display_id}...")
    try:
        ee.download(display_id, output_dir=output_dir)
        print(f"✅ Successfully downloaded: {display_id}")
    except Exception as e:
        print(f"❌ Error downloading {display_id}: {e}")

# Logout
ee.logout()
print("Download complete!")

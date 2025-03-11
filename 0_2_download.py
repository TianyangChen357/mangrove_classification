import requests
import os
import time
import datetime
import threading
import json
import re
import sys

'''
This script downloads Landsat scenes using their Entity IDs from USGS Earth Explorer.

python 0_2_download.py <#thread> <scene list file>

python 0_2_download.py 10 scene_list_entity_id.txt

'''


# Set download path
download_path = "0_landsat_downloads/"
os.makedirs(download_path, exist_ok=True)

# Set max concurrent downloads
max_threads = int(sys.argv[1])
sema = threading.Semaphore(value=max_threads)
threads = []

# USGS M2M API Endpoints
service_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
LOGIN_URL = service_url + "login-token"
DOWNLOAD_OPTIONS_URL = service_url + "download-options"
DOWNLOAD_REQUEST_URL = service_url + "download-request"
DOWNLOAD_RETRIEVE_URL = service_url + "download-retrieve"
LOGOUT_URL = service_url + "logout"

# ?? Hardcoded USGS Credentials
USERNAME = "tchen19"
TOKEN = "L7n6RhXN2gvJUsK4_z@V9bneeSZ7FuZEppG06gAm7e7sDfRpDJXWAf@y@uGzxpjG"

# Function to send API requests
def send_request(url, data, api_key=None):
    headers = {"X-Auth-Token": api_key} if api_key else {}
    response = requests.post(url, json.dumps(data), headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        if response_data["errorCode"] is None:
            # print(f"? Request successful: {url.split('/')[-1]}")
            return response_data["data"]
        else:
            print(f"? API Error: {response_data['errorMessage']}")
            return None
    else:
        print(f"? HTTP Error {response.status_code} for {url}")
        return None

# Function to download a file
def download_file(url):
    sema.acquire()
    try:
        response = requests.get(url, stream=True)
        disposition = response.headers.get("content-disposition", "")
        filename = re.findall(r"filename=(.+)", disposition)[0].strip("\"")
        file_path = os.path.join(download_path, filename)

        # print(f"? Downloading {filename} ...")
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        # print(f"? Downloaded: {filename}")

    except Exception as e:
        print(f"? Failed to download {url}. Retrying...")
        run_download(url)
    finally:
        sema.release()

# Function to start a download in a thread
def run_download(url):
    thread = threading.Thread(target=download_file, args=(url,))
    threads.append(thread)
    thread.start()

# Main Execution
if __name__ == "__main__":
    # Step 1: Authenticate and get API Key
    # print("\n?? Logging in to USGS M2M API...")
    api_key = send_request(LOGIN_URL, {"username": USERNAME, "token": TOKEN})
    if not api_key:
        print("? Authentication failed. Exiting...")
        exit()

    # print(f"? API Key: {api_key}\n")

    # Step 2: Read scene IDs from file
    scene_list_file = sys.argv[2]
    with open(scene_list_file, "r") as f:
        scene_ids = [line.strip() for line in f.readlines() if line.strip()]

    # print(f"?? {len(scene_ids)} scenes found in {scene_list_file}\n")

    # Step 3: Request download options
    # print("?? Finding available download options...")
    payload = {"datasetName": "landsat_tm_c2_l2", "entityIds": scene_ids}
    download_options = send_request(DOWNLOAD_OPTIONS_URL, payload, api_key)

    # Step 4: Filter available products
    downloads = []
    if download_options:
        for product in download_options:
            if product.get("available", False):
                downloads.append({"entityId": product["entityId"], "productId": product["id"]})

    if not downloads:
        print("? No valid downloads found. Exiting...")
        exit()

    # Step 5: Request downloads
    label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    request_payload = {"downloads": downloads, "label": label}
    request_results = send_request(DOWNLOAD_REQUEST_URL, request_payload, api_key)

    if not request_results:
        print("? No downloads available at the moment. Try again later.")
        exit()

    # Step 6: Retrieve download URLs
    download_ids = []
    if "preparingDownloads" in request_results and request_results["preparingDownloads"]:
        # print("\n? Waiting 30 seconds for downloads to become available...")

        retrieve_payload = {"label": label}
        more_download_urls = send_request(DOWNLOAD_RETRIEVE_URL, retrieve_payload, api_key)

        for download in more_download_urls.get("available", []):
            if str(download["downloadId"]) in request_results.get("newRecords", []) or str(download["downloadId"]) in request_results.get("duplicateProducts", []):
                download_ids.append(download["downloadId"])
                run_download(download["url"])

        for download in more_download_urls.get("requested", []):
            if str(download["downloadId"]) in request_results.get("newRecords", []) or str(download["downloadId"]) in request_results.get("duplicateProducts", []):
                download_ids.append(download["downloadId"])
                run_download(download["url"])

    else:
        # If downloads are already available
        for download in request_results.get("availableDownloads", []):
            run_download(download["url"])

    # Step 7: Wait for all downloads to finish
    # print("\n?? Downloading files... Please wait.\n")
    for thread in threads:
        thread.join()

    print("? All downloads completed!")

    # Step 8: Logout
    send_request(LOGOUT_URL, {}, api_key)
    # print("? Successfully logged out.")

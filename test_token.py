
import json
import requests
import sys
import time
import argparse
import datetime
import threading
import re

path = "" # Fill a valid path to save the downloaded files
maxthreads = 5 # Threads count for downloads
sema = threading.Semaphore(value=maxthreads)
label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Customized label using date time
threads = []

# send http request
def sendRequest(url, data, apiKey = None):  
    pos = url.rfind('/') + 1
    endpoint = url[pos:]
    json_data = json.dumps(data)
    
    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {'X-Auth-Token': apiKey}              
        response = requests.post(url, json_data, headers = headers)    
    
    try:
      httpStatusCode = response.status_code 
      if response == None:
          print("No output from service")
          sys.exit()
      output = json.loads(response.text)	
      if output['errorCode'] != None:
          print("Failed Request ID", output['requestId'])
          print(output['errorCode'], "-", output['errorMessage'])
          sys.exit()
      if  httpStatusCode == 404:
          print("404 Not Found")
          sys.exit()
      elif httpStatusCode == 401: 
          print("401 Unauthorized")
          sys.exit()
      elif httpStatusCode == 400:
          print("Error Code", httpStatusCode)
          sys.exit()
    except Exception as e: 
          response.close()
          pos=serviceUrl.find('api')
          print(f"Failed to parse request {endpoint} response. Re-check the input {json_data}. The input examples can be found at {url[:pos]}api/docs/reference/#{endpoint}\n")
          sys.exit()
    response.close()    
    print(f"Finished request {endpoint} with request ID {output['requestId']}\n")
    
    return output['data']

def downloadFile(url):
    sema.acquire()
    global path
    try:        
        response = requests.get(url, stream=True)
        disposition = response.headers['content-disposition']
        filename = re.findall("filename=(.+)", disposition)[0].strip("\"")
        print(f"Downloading {filename} ...\n")
        if path != "" and path[-1] != "/":
            filename = "/" + filename
        open(path + filename, 'wb').write(response.content)
        print(f"Downloaded {filename}\n")
        sema.release()
    except Exception as e:
        print(f"Failed to download from {url}. {e}. Will try to re-download.")
        sema.release()
        runDownload(threads, url)
    
def runDownload(threads, url):
    thread = threading.Thread(target=downloadFile, args=(url,))
    threads.append(thread)
    thread.start()

if __name__ == '__main__': 
    #NOTE :: Passing credentials over a command line arguement is not considered secure
    #        and is used only for the purpose of being example - credential parameters
    #        should be gathered in a more secure way for production usage
    #Define the command line arguements
    
    # user input    
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username', help='ERS Username')
    parser.add_argument('-t', '--token',  help='ERS application token')
    
    args = parser.parse_args()
    
    username = "tchen19"
    token = "L7n6RhXN2gvJUsK4_z@V9bneeSZ7FuZEppG06gAm7e7sDfRpDJXWAf@y@uGzxpjG"    

    print("\nRunning Scripts...\n")
    
    serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    
    # login-token
    payload = {'username' : username, 'token' : token}
    
    apiKey = sendRequest(serviceUrl + "login-token", payload)
    
    print("API Key: " + apiKey + "\n")
    
    ########################### dataset search####################
    datasetName = "landsat_tm_c2_l2"
    
    spatialFilter =  {'filterType' : "mbr",
                      'lowerLeft' : {'latitude' : 19.427111111111685, 'longitude' : -87.5246666},
                      'upperRight' : { 'latitude' : 19.427333333333905, 'longitude' : -87.52444444444492}}
                   #-87.52466666666714, 19.427111111111685, -87.52444444444492, 19.427333333333905  
    acquisitionFilter = {'start' : "2007-01-01", 'end' : "2010-12-31"}
    
       
    payload = {'datasetName' : datasetName, 
                                'maxResults' : 2,
                                'startingNumber' : 1, 
                                'sceneFilter' : {
                                                'spatialFilter' : spatialFilter,
                                                'acquisitionFilter' : acquisitionFilter}}
    
    # Now I need to run a scene search to find data to download
    print("Searching scenes...\n\n")   
    
    scenes = sendRequest(serviceUrl + "scene-search", payload, apiKey)

    # Did we find anything?
    if scenes['recordsReturned'] > 0:
        # Aggregate a list of scene ids
        sceneIds = []
        for result in scenes['results']:
            # Add this scene to the list I would like to download
            sceneIds.append(result['entityId'])
        
        # Find the download options for these scenes
        # NOTE :: Remember the scene list cannot exceed 50,000 items!
        payload = {'datasetName' : datasetName, 'entityIds' : sceneIds}
                            
        downloadOptions = sendRequest(serviceUrl + "download-options", payload, apiKey)
    
        # Aggregate a list of available products
        downloads = []
        for product in downloadOptions:
                # Make sure the product is available for this scene
                if product['available'] == True:
                        downloads.append({'entityId' : product['entityId'],
                                        'productId' : product['id']})
                        
        # Did we find products?
        if downloads:
            requestedDownloadsCount = len(downloads)
            # set a label for the download request
            label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Customized label using date time
            payload = {'downloads' : downloads,
                                            'label' : label}
            # Call the download to get the direct download urls
            requestResults = sendRequest(serviceUrl + "download-request", payload, apiKey)          
                            
            # PreparingDownloads has a valid link that can be used but data may not be immediately available
            # Call the download-retrieve method to get download that is available for immediate download
            if requestResults['preparingDownloads'] != None and len(requestResults['preparingDownloads']) > 0:
                payload = {'label' : label}
                moreDownloadUrls = sendRequest(serviceUrl + "download-retrieve", payload, apiKey)
                
                downloadIds = []  
                
                for download in moreDownloadUrls['available']:
                    if str(download['downloadId']) in requestResults['newRecords'] or str(download['downloadId']) in requestResults['duplicateProducts']:
                        downloadIds.append(download['downloadId'])
                        runDownload(threads, download['url'])
                    
                for download in moreDownloadUrls['requested']:
                    if str(download['downloadId']) in requestResults['newRecords'] or str(download['downloadId']) in requestResults['duplicateProducts']:
                        downloadIds.append(download['downloadId'])
                        runDownload(threads, download['url'])
                    
                # Didn't get all of the reuested downloads, call the download-retrieve method again probably after 30 seconds
                while len(downloadIds) < (requestedDownloadsCount - len(requestResults['failed'])): 
                    preparingDownloads = requestedDownloadsCount - len(downloadIds) - len(requestResults['failed'])
                    print("\n", preparingDownloads, "downloads are not available. Waiting for 30 seconds.\n")
                    time.sleep(30)
                    print("Trying to retrieve data\n")
                    moreDownloadUrls = sendRequest(serviceUrl + "download-retrieve", payload, apiKey)
                    for download in moreDownloadUrls['available']:                            
                        if download['downloadId'] not in downloadIds and (str(download['downloadId']) in requestResults['newRecords'] or str(download['downloadId']) in requestResults['duplicateProducts']):
                            downloadIds.append(download['downloadId'])
                            runDownload(threads, download['url'])
                        
            else:
                # Get all available downloads
                for download in requestResults['availableDownloads']:
                    runDownload(threads, download['url'])

    
    print("Downloading files... Please do not close the program\n")
    for thread in threads:
        thread.join()
            
    print("Complete Downloading")
                
    # Logout so the API Key cannot be used anymore
    endpoint = "logout"  
    if sendRequest(serviceUrl + endpoint, None, apiKey) == None:        
        print("Logged Out\n\n")
    else:
        print("Logout Failed\n\n")            
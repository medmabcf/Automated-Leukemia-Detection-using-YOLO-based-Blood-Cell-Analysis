import os
import requests

# The URL of the file you want to download
url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

# The path where you want to save the downloaded file
save_path = './models/sam_vit_h_4b8939.pth'

# Check if the file already exists
if not os.path.exists(save_path):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in write mode
        with open(save_path, 'wb') as file:
            # Write the contents of the response to the file
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
else:
    print("The file already exists.")

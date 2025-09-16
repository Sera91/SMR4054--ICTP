import requests
import zipfile
import sys
from tqdm import tqdm

Toubkal_username=str(sys.argv[1])

domain = 'NZ'
DATA_PATH = '/home/'+ Toubkal_username+'/lustre/training_t-7bpcjwtiufg/'+ Toubkal_username+'/SMR4054--ICTP/Tutorial/Day7/RCM-emulation-tutorial/data/CORDEXBench/{domain}'
os.makedirs(DATA_PATH, exist_ok=True)


def download_and_extract(domain, DATA_PATH=DATA_PATH):
    BASE_URL = "https://zenodo.org/records/15797226/files"

    zip_path = os.path.join(DATA_PATH, f"{domain}_domain.zip")
    download_url = f"{BASE_URL}/{domain}_domain.zip?download=1"

    # Stream download with progress bar
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))
        block_size = 8192  # 8 KB

        with open(zip_path, 'wb') as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Downloading {domain}_domain.zip"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=block_size):
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract zip contents into DATA_PATH
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)

    # Remove the zip file after extraction
    os.remove(zip_path)



download_and_extract(domain)


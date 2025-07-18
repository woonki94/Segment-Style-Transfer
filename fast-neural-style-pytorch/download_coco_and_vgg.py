import os
import requests
from tqdm import tqdm
import zipfile

# Base directory = directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Directories
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ----------------------------
# COCO Dataset (train2014)
# ----------------------------
coco_url = "http://images.cocodataset.org/zips/train2014.zip"
coco_zip_path = os.path.join(dataset_dir, "train2014.zip")
coco_extract_path = os.path.join(dataset_dir, "train2014")

def download_file(url, dest_path, desc=""):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as file, tqdm(
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc=desc
    ) as bar:
        for data in response.iter_content(1024):
            file.write(data)
            bar.update(len(data))

# Download COCO train2014.zip
if not os.path.exists(coco_zip_path):
    download_file(coco_url, coco_zip_path, "Downloading COCO Train2014")
else:
    print("COCO zip already exists, skipping download.")

# Extract COCO if not already extracted
if not os.path.exists(coco_extract_path):
    print("Extracting COCO train2014.zip...")
    with zipfile.ZipFile(coco_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("COCO extraction complete.")
else:
    print("COCO already extracted, skipping.")

# ----------------------------
# VGG16 Model
# ----------------------------
vgg_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
vgg_path = os.path.join(models_dir, "vgg16-397923af.pth")

# Download VGG16 weights
if not os.path.exists(vgg_path):
    download_file(vgg_url, vgg_path, "Downloading VGG16 Weights")
else:
    print("VGG16 weights already exist, skipping download.")

# Final paths summary
print(f"\nCOCO train2014 folder: {coco_extract_path}")
print(f" VGG16 weights path: {vgg_path}")


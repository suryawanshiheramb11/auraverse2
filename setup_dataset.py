import os
import cv2
import numpy as np
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataGen")

DATASET_DIR = "dataset"
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

def ensure_dirs():
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

def download_image(index):
    try:
        # Get a random image (224x224)
        url = f"https://picsum.photos/224?random={index}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            path = os.path.join(REAL_DIR, f"real_{index}.jpg")
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
    except Exception as e:
        logger.warning(f"Failed to download image {index}: {e}")
    return None

def apply_artifacts(image_path, index):
    try:
        img = cv2.imread(image_path)
        if img is None: return
        
        # Artifact 1: Gaussian Blur (Smoothing effect common in bad deepfakes)
        augmented = cv2.GaussianBlur(img, (7, 7), 3)
        
        # Artifact 2: JPEG Compression Noise
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, encimg = cv2.imencode('.jpg', augmented, encode_param)
        augmented = cv2.imdecode(encimg, 1)
        
        # Artifact 3: Slight Color Shift
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 20) # Brightness/Glow
        final_hsv = cv2.merge((h, s, v))
        augmented = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
        fake_path = os.path.join(FAKE_DIR, f"fake_{index}.jpg")
        cv2.imwrite(fake_path, augmented)
        
    except Exception as e:
        logger.warning(f"Failed to generate fake {index}: {e}")

def main():
    logger.info("Generating Synthetic Dataset for Training...")
    ensure_dirs()
    
    count = 50 # Generate 50 pairs
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        logger.info(f"Downloading {count} base images...")
        paths = list(executor.map(download_image, range(count)))
        
    logger.info("Generating 'Fake' samples via augmentation...")
    for i, path in enumerate(paths):
        if path:
            apply_artifacts(path, i)
            
    logger.info(f"Dataset Ready: {count} Real, {count} Fake samples.")
    logger.info(f"Location: {os.path.abspath(DATASET_DIR)}")
    logger.info("You can now run 'python3 train_model.py'")

if __name__ == "__main__":
    main()

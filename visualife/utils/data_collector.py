import os
import time
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image
import random

# =========================
# Configuration
# =========================
classes = ["chair", "table", "person", "door", "steps", "car", "laptop", "cycle", "book", "phone", "bike"]
num_images_per_class = 5000
img_size = (128, 128)
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# =========================
# Function to download images from Google and Bing
# =========================
def download_images(keyword, num_images, storage_dir):
    # Count already downloaded images
    existing_images = len(os.listdir(storage_dir))
    remaining = num_images - existing_images
    if remaining <= 0:
        print(f"Skipping '{keyword}' — already have {existing_images} images.")
        return

    # Split remaining images between Google and Bing
    per_source = remaining // 2 + remaining % 2  # distribute extra to Google

    print(f"Downloading {remaining} images for class '{keyword}' (Google + Bing)...")

    # Google
    google_crawler = GoogleImageCrawler(storage={'root_dir': storage_dir})
    google_crawler.crawl(keyword=keyword, max_num=per_source)

    # Bing
    bing_crawler = BingImageCrawler(storage={'root_dir': storage_dir})
    bing_crawler.crawl(keyword=keyword, max_num=remaining - per_source)

    print(f"Finished downloading images for class '{keyword}'")

# =========================
# Download & clean dataset
# =========================
for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    os.makedirs(class_dir, exist_ok=True)

    download_images(cls, num_images_per_class, class_dir)
    time.sleep(random.uniform(0.5, 1))   # small delay to avoid blocking

# =========================
# Resize & clean images
# =========================
print("Resizing and cleaning images...")

for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(img_size)
                img.save(img_path)
        except Exception as e:
            print(f"Removing corrupt image: {img_path} -> {e}")
            os.remove(img_path)

print("Dataset is ready!")

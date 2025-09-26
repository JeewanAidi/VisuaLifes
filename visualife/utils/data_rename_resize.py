import os
from PIL import Image

# Paths
dataset_dir = "dataset"  
output_dir = "dataset_resized"  # new folder for resized dataset
classes = ["chair", "table", "person", "door", "stairs", "vehicle", "laptop", "bottle", "phone"]

# Desired image size
img_size = (128, 128)

# Create output dataset root
os.makedirs(output_dir, exist_ok=True)

for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    out_class_dir = os.path.join(output_dir, cls)
    os.makedirs(out_class_dir, exist_ok=True)

    print(f"Processing class: {cls}")

    if not os.path.exists(class_dir):
        print(f"⚠️ Skipping {cls}, folder not found.")
        continue

    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

    for idx, filename in enumerate(files):
        old_path = os.path.join(class_dir, filename)
        new_name = f"{cls}_{idx+1:05d}.jpg"
        new_path = os.path.join(out_class_dir, new_name)

        try:
            with Image.open(old_path) as img:
                img = img.convert("RGB")
                img = img.resize(img_size)
                img.save(new_path, "JPEG")  # save resized image in new dataset

        except Exception as e:
            print(f"❌ Error with {old_path}: {e}")

    print(f"✅ Finished {cls}, saved in '{out_class_dir}'")

print("🎉 All images resized & renamed! Original dataset is untouched.")

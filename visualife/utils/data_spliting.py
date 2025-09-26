import os
import shutil
import random

dataset_dir = "dataset_resized"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
test_ratio = 0.2

classes = ["chair", "table", "person", "door", "stairs", "vehicle", "laptop", "bottle", "phone"]

for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(files)
    n_test = int(len(files) * test_ratio)

    # Create folders
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Move files
    for f in files[n_test:]:
        shutil.move(os.path.join(class_dir, f), os.path.join(train_dir, cls, f))
    for f in files[:n_test]:
        shutil.move(os.path.join(class_dir, f), os.path.join(test_dir, cls, f))

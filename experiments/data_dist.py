# Check your training data distribution
import os
from collections import Counter

def check_class_distribution(data_dir):
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
            class_counts[class_name] = num_images
            print(f"{class_name}: {num_images} images")
    
    return class_counts

# Run this check
data_dir = "dataset_resized/train"
class_distribution = check_class_distribution(data_dir)
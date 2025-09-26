# visualife/utils/data_loader.py
import os
import numpy as np
from PIL import Image
import random

class DataLoader:
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32,
                 num_classes=None, shuffle=True):
        """
        data_dir: folder path with subfolders as class names
        img_size: (width, height) for resizing
        batch_size: number of images per batch
        num_classes: number of classes (auto-detected if None)
        shuffle: shuffle dataset every epoch
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load all file paths
        self.class_names = sorted(os.listdir(data_dir))
        self.num_classes = num_classes or len(self.class_names)

        self.samples = []
        for idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, fname), idx))

        print(f"📂 Found {len(self.samples)} images in {len(self.class_names)} classes: {self.class_names}")

        self.index = 0
        if self.shuffle:
            random.shuffle(self.samples)

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            random.shuffle(self.samples)
        return self

    def __next__(self):
        if self.index >= len(self.samples):
            raise StopIteration

        batch_samples = self.samples[self.index:self.index + self.batch_size]
        X, y = [], []

        for filepath, label in batch_samples:
            try:
                img = Image.open(filepath).convert("RGB")
                img = img.resize(self.img_size)
                X.append(np.array(img) / 255.0)
                y_vec = np.zeros(self.num_classes)
                y_vec[label] = 1
                y.append(y_vec)
            except Exception as e:
                print(f"⚠️ Skipping {filepath}, error: {e}")

        self.index += self.batch_size

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return X, y

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

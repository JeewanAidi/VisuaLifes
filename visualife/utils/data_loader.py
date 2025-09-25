# visualife/utils/data_loader.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dataset_path, img_size=(128, 128), test_ratio=0.2):
        """
        dataset_path: path to dataset folder (subfolders = class names)
        img_size: (width, height) for resizing images
        test_ratio: fraction of data to use for testing
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.test_ratio = test_ratio

        # automatically get class names from folder names
        self.class_names = sorted(os.listdir(dataset_path))
        self.num_classes = len(self.class_names)
        print(f"Found classes: {self.class_names}")

    def load_data(self):
        X, y = [], []

        for idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                try:
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize(self.img_size)
                    X.append(np.array(img) / 255.0)  # normalize
                    y.append(idx)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        # One-hot encode labels
        y_onehot = np.zeros((y.shape[0], self.num_classes))
        y_onehot[np.arange(y.shape[0]), y] = 1

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=self.test_ratio, random_state=42, shuffle=True
        )

        print(f"Total images: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, y_train, X_test, y_test

# tests/test_data_loader.py
import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import numpy as np
from visualife.utils.data_loader import DataLoader
from PIL import Image

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create dummy dataset
        self.dataset_path = 'tests/dummy_dataset'
        os.makedirs(self.dataset_path, exist_ok=True)
        self.classes = ['chair', 'door', 'person']
        for cls in self.classes:
            cls_path = os.path.join(self.dataset_path, cls)
            os.makedirs(cls_path, exist_ok=True)
            # create 3 dummy images per class
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
                img.save(os.path.join(cls_path, f"{cls}_{i}.png"))

        self.loader = DataLoader(self.dataset_path, img_size=(128, 128), test_ratio=0.2)

    def tearDown(self):
        # Remove dummy dataset after test
        shutil.rmtree(self.dataset_path)

    def test_load_data_shapes(self):
        X_train, y_train, X_test, y_test = self.loader.load_data()
        self.assertEqual(X_train.shape[1:], (128, 128, 3))
        self.assertEqual(X_test.shape[1:], (128, 128, 3))
        self.assertEqual(y_train.shape[1], len(self.classes))
        self.assertEqual(y_test.shape[1], len(self.classes))
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test:", X_test.shape, "y_test:", y_test.shape)

if __name__ == "__main__":
    unittest.main()

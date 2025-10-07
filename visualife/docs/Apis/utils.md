#  Utils Module — Data & Helper Functions

> Utility functions and data tools used across the VisuaLife framework.

---

##  Overview

The `visualife.utils` package provides essential helper utilities to manage datasets, perform preprocessing, and assist with visualization or experimentation tasks.

These tools help you:
- Load custom datasets  
- Split and preprocess images  
- Collect statistics from datasets  
- Manage experiment data and logs

---

##  Import

```python
from visualife.utils.data_loader import DataLoader
from visualife.utils.data_collector import DataCollector
```

##  Class: DataLoader
Efficiently loads image datasets from folders, automatically organizing them by class and providing them in batches for training.

**Initialization**
```python
loader = DataLoader(
    data_dir="dataset/",
    img_size=(128, 128),
    batch_size=32,
    num_classes=None,
    shuffle=True
)
```
**Arguments**
| Argument    | Type   | Description                                      |
|------------|--------|--------------------------------------------------|
| data_dir    | str    | Path to dataset folder containing subfolders per class |
| img_size    | tuple  | Image resize size (width, height)               |
| batch_size  | int    | Number of images per batch                       |
| num_classes | int    | Optional number of classes (auto-detected if None) |
| shuffle     | bool   | Shuffle dataset each epoch                       |


**Usage Example**
```python
from visualife.utils.data_loader import DataLoader

loader = DataLoader("dataset/", img_size=(64, 64), batch_size=16)

for X, y in loader:
    print(X.shape, y.shape)
    # Train your model here

```
**Output Example:**

```less
📂 Found 500 images in 5 classes: ['cat', 'dog', 'car', 'tree', 'person']
(16, 64, 64, 3) (16, 5)
```

##  Class: DataCollector
Used for collecting, visualizing, and saving experiment metrics such as accuracy and loss curves.

**Initialization**
```python
collector = DataCollector()
```
**Methods**
| Method       | Description                                    |
|-------------|------------------------------------------------|
| log(metric_name, value) | Log a single metric (e.g., accuracy, loss) |
| plot_metrics()           | Visualize all collected metrics using Matplotlib |
| save(path)               | Save metrics as a .pkl or .csv file |


**Example**
```python
from visualife.utils.data_collector import DataCollector

collector = DataCollector()
collector.log("accuracy", 0.87)
collector.log("loss", 0.32)

collector.plot_metrics()
collector.save("results/metrics.pkl")
```

##  Tips

- Combine **DataLoader** + **DataCollector** to manage the full experiment pipeline.
- Works with custom datasets for CNNs, MLPs, or any deep model.
- Automatically normalizes images (0–1 range).


##  Summary

| Utility        | Purpose                                           |
|----------------|--------------------------------------------------|
| DataLoader     | Handles dataset loading and preprocessing       |
| DataCollector  | Stores and visualizes training logs and metrics |


##  Next

[nlp_training_mnist.md →](nlp_training_mnist.md)

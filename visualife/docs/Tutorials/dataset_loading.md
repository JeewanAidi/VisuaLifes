# Dataset Loading with VisuaLife

> Learn how to load and preprocess datasets for training neural networks using VisuaLife’s utilities.

---

##  Objective

By the end of this tutorial, you will be able to:

- Load built-in datasets like CIFAR-10  
- Preprocess images and labels  
- Split data into training and test sets  
- Feed data to your model for training

---

##  1. Import Modules

```python
from visualife.utils.data_loader import load_cifar10
import numpy as np
```

##  2. Load CIFAR-10 Dataset
```python
# Returns: (X_train, y_train), (X_test, y_test)
(X_train, y_train), (X_test, y_test) = load_cifar10()
```
X_train.shape → `(50000, 32, 32, 3)`  
y_train.shape → `(50000, 10)`  
X_test.shape → `(10000, 32, 32, 3)`  
y_test.shape → `(10000, 10)`


##  3. Preprocessing Data
VisuaLife expects float inputs normalized between 0 and 1:

```python
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```
Labels are one-hot encoded (already done in load_cifar10).

##  4. Feed Data to the Model
```python
from visualife.core.model import Model
from visualife.core.layers import Dense, Flatten, Activation
from visualife.core.losses import CrossEntropyLoss
from visualife.core.optimizers import SGD

model = Model([
    Flatten(),            # Flatten 32x32x3 → 3072
    Dense(3072, 128),
    Activation('relu'),
    Dense(128, 10),
    Activation('softmax')
])

model.compile(
    optimizer=SGD(lr=0.01),
    loss=CrossEntropyLoss()
)

model.fit(X_train, y_train, epochs=5)
```
For demonstration, we use 5 epochs. Increase epochs for higher accuracy.

##  5. Evaluate on Test Set
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
```

## 6. Summary

You have learned to:

- **Load CIFAR-10** with `data_loader`
- **Preprocess** inputs and labels
- **Feed data** into your model
- **Train and evaluate** on real datasets

This is the foundation for all image classification tasks with VisuaLife.

## Next Steps

- [**Train a complete model →**](training.md) See `training.md`  
- [**Explore API Reference →**](Apis/layers.md) Check available layers and functions  
- [**Try advanced examples →**](nlp_training_mnist.md) See `nlp_training_mnist.md`



© 2025 VisuaLife | Written by Jeewan Aidi
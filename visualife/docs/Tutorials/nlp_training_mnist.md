#  MLP Training on MNIST — With VisuaLife

> Train a simple **Multi-Layer Perceptron (MLP)** on the MNIST handwritten digit dataset using the **VisuaLife** deep learning framework.

---

##  Objective

By the end of this tutorial, you will learn to:

- Load and preprocess the **MNIST dataset**
- Build a neural network using **VisuaLife layers**
- Compile, train, and evaluate the model
- Visualize performance results

---

##  1. Import Modules

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from visualife.core.model import Model
from visualife.core.layers import Dense
from visualife.core.activations import ReLU, Softmax
from visualife.core.losses import CrossEntropyLoss
from visualife.core.optimizers import Adam
```

##  2. Load Dataset
```python
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Normalize pixel values
X = X / 255.0
y = y.astype(int)

# One-hot encode labels
num_classes = 10
Y = np.eye(num_classes)[y]

# Split dataset for faster training (10k train / 2k test)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[:12000], Y[:12000], test_size=2000, random_state=42)
```

##  3. Build the Model
```python
print("Building model...")
model = Model()

model.add(Dense(784, 128))
model.add(ReLU())
model.add(Dense(128, 64))
model.add(ReLU())
model.add(Dense(64, 10))
model.add(Softmax())
```

##  4. Compile Model
```python
loss = CrossEntropyLoss()
optimizer = Adam(learning_rate=0.001)
model.compile()
```

##  5. Train the Model
```python
print("Training model...")
model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, Y_test)
)
```

##  6. Evaluate Model
```python
print("Evaluating model...")
preds = model.predict(X_test)
acc = np.mean(np.argmax(preds, axis=1) == np.argmax(Y_test, axis=1))
print(f"Final Test Accuracy: {acc:.4f}")
```
**Output Example**
```yaml
 Loading MNIST dataset...
 Building model...
 Training model...
Epoch [100/100] - Loss: 0.0253 - Val_Loss: 0.0407 - Val_Acc: 98.40%
Final Test Accuracy: 0.9843
```

## Key Takeaways

- VisuaLife provides a clean layer-based architecture for building neural networks.
- The `Model` class works like `Sequential` in Keras.
- You can easily replace layers and optimizers to experiment with different architectures.


##  Next

[contributing.md →](contributing.md)

© 2025 VisuaLife | Written by Jeewan Aidi
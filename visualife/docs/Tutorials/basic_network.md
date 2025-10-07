# Basic Neural Network Tutorial

> Learn how to create, train, and evaluate your first neural network using **VisuaLife**.

---

##  Objective

We will train a **simple feed-forward neural network** on a small dataset (XOR) to demonstrate:

- Building a network
- Forward and backward passes
- Loss computation
- Weight updates
- Making predictions

This is a simplified example before moving to larger datasets like CIFAR-10.

---

##  1. Import Modules

```python
from visualife.core.model import Model
from visualife.core.layers import Dense, Activation
from visualife.core.losses import MeanSquaredError
from visualife.core.optimizers import SGD
import numpy as np
```

##  2. Prepare Dataset (XOR)
```python
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])
```

##  3. Build the Model
```python
model = Model([
    Dense(2, 4),          # Input: 2 → Hidden: 4
    Activation('relu'),
    Dense(4, 1),          # Hidden: 4 → Output: 1
    Activation('sigmoid')
])
```

##  4. Compile the Model
```python
model.compile(
    optimizer=SGD(lr=0.1),
    loss=MeanSquaredError()
)
```

##  5. Train the Model
```python
model.fit(X, y, epochs=500)
```
During training, you’ll see the loss decrease as the model learns the XOR pattern.

##  6. Evaluate / Make Predictions
```python
predictions = model.predict(X)
print("Predictions:\n", predictions)
```
Expected output:

```lua
Predictions:
[[0.01]
 [0.98]
 [0.97]
 [0.02]]

```
Values close to [0, 1, 1, 0] — success!

##  7. Summary

You have learned to:

- Prepare data
- Build a neural network with layers and activations
- Compile the model with optimizer and loss
- Train the model with backpropagation
- Make predictions

This is the first step toward more complex tasks like image classification.


## Next Steps

- [**Training Tutorial →**](nlp_training_mnist.md) Learn more about epochs, batch sizes, and optimizers  
- [**Dataset Loading →**](Apis/utils.md) How to load CIFAR-10 and other datasets  
- [**API Reference →**](Apis/layers.md) Explore all available classes, layers, and functions in VisuaLife



© 2025 VisuaLife | Written by Jeewan Aidi
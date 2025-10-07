# Getting Started with VisuaLife

> A hands-on introduction to building, training, and testing your first neural network using VisuaLife.

---

##  What You’ll Learn

By the end of this tutorial, you’ll know how to:

- Create a neural network using **VisuaLife’s modular layers**
- Load and preprocess sample data
- Train the network on a small dataset
- Evaluate its performance

---

##  1. Import Core Modules

Let’s begin by importing the main classes from VisuaLife:

```python
from visualife.core.model import Model
from visualife.core.layers import Dense, Activation
from visualife.core.losses import MeanSquaredError
from visualife.core.optimizers import SGD
```

##  2. Build a Simple Neural Network
Here’s an example of a small feed-forward network with two hidden layers:

```python
model = Model([
    Dense(2, 4),          # Input layer: 2 features → 4 neurons
    Activation('relu'),
    Dense(4, 1),          # Output layer: 1 neuron
    Activation('sigmoid')
])
```
This defines your network architecture — VisuaLife takes care of forward and backward propagation internally.

##  3. Prepare Sample Data
Let’s train it on a tiny XOR dataset for simplicity:

```python
import numpy as np

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

##  4. Compile and Train the Model
Now, configure the optimizer and loss function:

```python
model.compile(
    optimizer=SGD(lr=0.1),
    loss=MeanSquaredError()
)

model.fit(X, y, epochs=500)
```
During training, you’ll see the loss decreasing over epochs.

##  5. Test the Model
After training, try predicting new values:

```python
predictions = model.predict(X)
print(predictions)
```
You should see outputs close to [0, 1, 1, 0].

##  Summary
You just:
- Created a model
- Added layers
- Trained it on XOR data
- Made predictions

This is your first working neural network using VisuaLife built from scratch!

## Next Steps

- [Learn about Architecture →](Architecture.md)
- [Explore Tutorials →](Tutorials/basic_network.md)
- [Check API References →](Apis/layers.md)


© 2025 VisuaLife | Created by Jeewan Aidi
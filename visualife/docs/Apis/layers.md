# Layers API Reference

> Detailed reference for all core layer classes in **VisuaLife**.

---

## Overview

VisuaLife layers are the **building blocks** of your neural network.  
Every layer handles **forward** and **backward propagation**, stores parameters, and computes gradients.

Common layers:

| Layer | Description |
|-------|-------------|
| `Dense` | Fully connected layer |
| `Conv2D` | 2D convolutional layer |
| `Flatten` | Flatten multi-dimensional input |
| `Activation` | Apply activation function (ReLU, Sigmoid, etc.) |

---

## Dense Layer

```python
from visualife.core.layers import Dense

# Initialize a dense layer
dense = Dense(input_size=128, output_size=64)

```

**Attributes:**

- `weights` → Weight matrix
- `biases` → Bias vector

**Methods:**

- `forward(inputs)` → Computes output
- `backward(grad_output, learning_rate)` → Computes gradient and updates weights


## Conv2D Layer
```python
from visualife.core.layers import Conv2D

conv = Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1))
```

**Attributes:**

- `filters` → Number of convolution kernels
- `kernel_size` → Size of each kernel

**Methods:**

- `forward(inputs)` → Applies convolution
- `backward(grad_output, learning_rate)` → Backpropagates gradients

## Flatten Layer
```python
from visualife.core.layers import Flatten

flatten = Flatten()
```

**Purpose:** Converts multi-dimensional input to 1D vector.

**Methods:**

- `forward(inputs)` → Flattened output
- `backward(grad_output)` → Reshape gradient to original dimensions

## Activation Layer
```python
from visualife.core.layers import Activation

relu = Activation('relu')
sigmoid = Activation('sigmoid')
softmax = Activation('softmax')
```

**Purpose:** Apply non-linear activation functions.

**Supported Activations:**

- `'relu'` → Rectified Linear Unit
- `'sigmoid'` → Sigmoid function
- `'tanh'` → Hyperbolic tangent
- `'softmax'` → Probability distribution for classification

**Methods:**

- `forward(x)` → Compute activation
- `backward(grad_output)` → Compute gradient

## Summary

Layers in VisuaLife are modular and composable:

```python
from visualife.core.model import Model
from visualife.core.layers import Dense, Activation, Flatten

model = Model([
    Flatten(),          # Flatten input
    Dense(3072, 128),   # Fully connected
    Activation('relu'), # Non-linear
    Dense(128, 10),     # Output layer
    Activation('softmax') # Probabilities
])

```
This flexibility allows building any feed-forward or convolutional network.

## Next Steps

- Model API Reference →
- Utils API Reference →
- Try layers in Tutorials →

© 2025 VisuaLife | Written by Jeewan Aidi
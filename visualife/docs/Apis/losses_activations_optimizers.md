# Utils API Reference

> Utility functions and helper modules that power **VisuaLife** — including data loading, activation functions, losses, and optimizers.

---

##  Overview

The **utils** package in VisuaLife contains essential tools for:
- Activation functions  
- Loss functions  
- Optimizers (SGD, Adam, etc.)

This helps you build, train, and test deep learning models efficiently.

---

##  Activation Functions
Located in visualife/core/activations.py

Supported Activations
| Name    | Formula                 | Purpose                        |
|---------|------------------------|--------------------------------|
| relu    | f(x) = max(0, x)       | Introduces non-linearity       |
| sigmoid | 1 / (1 + e^-x)         | Binary classification          |
| tanh    | (e^x - e^-x)/(e^x + e^-x) | Normalized output [-1, 1] |
| softmax | e^x / Σe^x             | Multiclass classification      |

**Example:**
```python
from visualife.utils.activations import relu, softmax

x = np.array([-2, -1, 0, 1, 2])
print(relu(x))     # [0, 0, 0, 1, 2]
print(softmax(x))  # Normalized probabilities
```

##  Loss Functions
Located in visualife/core/losses.py

**categorical_crossentropy(y_true, y_pred)**
Computes the cross-entropy loss between true labels and predictions.

```python
from visualife.utils.losses import categorical_crossentropy

loss = categorical_crossentropy(y_true, y_pred)
```
| Parameter | Type     | Description                          |
|-----------|----------|--------------------------------------|
| y_true    | ndarray  | One-hot encoded ground truth labels  |
| y_pred    | ndarray  | Model output probabilities           |

Returns a scalar loss value.

##  Optimizers
Located in visualife/core/optimizers.py

### 1️⃣ Stochastic Gradient Descent (SGD)
```python
from visualife.utils.optimizers import SGD

optimizer = SGD(lr=0.01)
optimizer.update(params, grads)
```
| Parameter | Description                         |
|-----------|-------------------------------------|
| lr        | Learning rate                        |
| params    | Model parameters                     |
| grads     | Gradients from backpropagation       |


### 2️⃣ Adam Optimizer
```python
from visualife.utils.optimizers import Adam

optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
optimizer.update(params, grads)
```
| Parameter | Description                         |
|-----------|-------------------------------------|
| lr        | Learning rate                        |
| beta1     | Exponential decay rate for momentum  |
| beta2     | Exponential decay rate for RMS term  |


Adam combines the benefits of Momentum and RMSProp for faster convergence.


**Example Workflow**
```python
from visualife.utils.losses import categorical_crossentropy
from visualife.utils.optimizers import Adam
from visualife.utils.activations import relu


optimizer = Adam(lr=0.001)
loss = categorical_crossentropy(y_train[0], y_train[0])  # Example
print("Sample loss:", loss)
```

##  Summary
Module	Purpose
activations.py	Activation functions
losses.py	Loss functions
optimizers.py	Optimization algorithms

##  Next Steps
- [Model API Reference →](Apis/layers.md)
- [Layers API Reference →](Tutorials/basic_network.md)
- [nlp_training_mnist.md →](nlp_training_mnist.md)

© 2025 VisuaLife | Written by Jeewan Aidi
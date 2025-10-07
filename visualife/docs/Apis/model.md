# Model API Reference

> Core API documentation for the **Model** class in VisuaLife — the main interface for building, training, and evaluating neural networks.

---

##  Overview

The `Model` class in **VisuaLife** serves as the central hub of your deep learning pipeline.  
It manages:
- Layer stacking
- Forward propagation
- Backward propagation
- Parameter updates
- Training loop (epochs, loss, metrics)

A typical workflow:

```python
from visualife.core.model import Model
from visualife.core.layers import Dense, Activation, Flatten
from visualife.utils.losses import categorical_crossentropy
from visualife.utils.optimizers import SGD

# Define model
model = Model([
    Flatten(),
    Dense(3072, 128),
    Activation('relu'),
    Dense(128, 10),
    Activation('softmax')
])

# Compile model
model.compile(
    loss=categorical_crossentropy,
    optimizer=SGD(lr=0.01)
)

# Train model
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

##  Model Class
Initialization
```python
Model(layers: list)
```

Parameter	Type	Description
layers	list	List of layer objects (Dense, Conv2D, etc.) to build the network.

##  Core Methods

### 1️⃣ `compile(loss, optimizer)`

Prepares the model for training by setting the **loss function** and **optimizer**.

**Arguments:**

| Argument   | Description |
|-------------|-------------|
| `loss`      | Loss function (e.g., `categorical_crossentropy`) |
| `optimizer` | Optimizer instance (e.g., `SGD`, `Adam`) |


### 2️⃣ `fit(x_train, y_train, epochs, batch_size)`

Trains the model for a given number of **epochs**.

**Parameters:**

| Parameter   | Type     | Description |
|--------------|----------|-------------|
| `x_train`    | `ndarray` | Training data |
| `y_train`    | `ndarray` | Target labels |
| `epochs`     | `int`     | Number of epochs to train |
| `batch_size` | `int`     | Samples per gradient update |


**Example:**

```python
model.fit(x_train, y_train, epochs=5, batch_size=64)
```

### 3️⃣ `predict(x)`
Runs forward propagation to predict outputs.

```python
predictions = model.predict(x_test)
```

### 4️⃣ `evaluate(x_test, y_test)`
Evaluates model accuracy and loss on test data.

```python
loss, accuracy = model.evaluate(x_test, y_test)
```
**Returns:**

| Return     | Description              |
|------------|--------------------------|
| `loss`     | Final test loss          |
| `accuracy` | Computed accuracy metric |


### 5️⃣ `summary()`
Prints a table describing the **model architecture**.

**Example Output:**

```markdown
-----------------------------------------
Layer (type)        Output Shape    Params
=========================================
Flatten             (None, 3072)    0
Dense               (None, 128)     393344
Activation (relu)   (None, 128)     0
Dense               (None, 10)      1290
Activation (softmax)(None, 10)      0
=========================================
Total Params: 394,634
```

**Example: Full Workflow**
```python
from visualife.core.model import Model
from visualife.core.layers import Dense, Activation, Flatten
from visualife.utils.data_loader import load_cifar10
from visualife.utils.losses import categorical_crossentropy
from visualife.utils.optimizers import SGD

# Load data
(x_train, y_train), (x_test, y_test) = load_cifar10()

# Create model
model = Model([
    Flatten(),
    Dense(3072, 128),
    Activation('relu'),
    Dense(128, 10),
    Activation('softmax')
])

# Compile and train
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.01))
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate
model.evaluate(x_test, y_test)

# Predict
preds = model.predict(x_test[:5])
print(preds)
```

## Notes

- Each layer automatically tracks **gradients** for backpropagation.  
- Custom layers can be added by **inheriting from `BaseLayer`**.  
- The training process is fully **NumPy-based** — no TensorFlow or PyTorch dependency.


## Next Steps

- [Layers API Reference →](Apis/layers.md)
- [Utils API Reference →](Apis/utils.md)
- [Losses Activation and Optimizers →](Apis/losses_activation_optimizers.md)
- [Tutorial →](Tutorials/basic_network.md)


© 2025 VisuaLife | Written by Jeewan Aidi
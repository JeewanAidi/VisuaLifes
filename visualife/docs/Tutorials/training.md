# Training Your Neural Network

> Learn how to train your model effectively using VisuaLife’s training pipeline.

---

##  Objective

In this tutorial, you will understand:

- The role of **epochs**  
- The concept of **batch size**  
- How **optimizers** update weights  
- Monitoring **loss and metrics** during training

This knowledge helps you train **larger networks** on real datasets.

---

##  1. Import Modules

```python
from visualife.core.model import Model
from visualife.core.layers import Dense, Activation
from visualife.core.losses import MeanSquaredError
from visualife.core.optimizers import SGD
import numpy as np
```

##  2. Prepare Sample Data
We will continue using the XOR dataset:

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

##  3. Build and Compile the Model
```python
model = Model([
    Dense(2, 4),
    Activation('relu'),
    Dense(4, 1),
    Activation('sigmoid')
])

model.compile(
    optimizer=SGD(lr=0.1),
    loss=MeanSquaredError()
)
```

## 4. Understand Training Parameters
- `Epochs`: Number of times the entire dataset is passed through the network.
- `Batch Size`: Number of samples processed before the weights are updated (VisuaLife uses full-batch by default).
- `Learning Rate (lr)`: Determines how big each weight update step is.

## 5. Train the Model
```python
model.fit(X, y, epochs=500)
```
During training, VisuaLife prints:
```yaml
Epoch 1/500 - Loss: 0.34
Epoch 2/500 - Loss: 0.28
...
Epoch 500/500 - Loss: 0.01
```
The loss decreases as the network learns.

## 6. Visualizing Training (Optional)
You can plot loss over epochs using matplotlib:

```python
import matplotlib.pyplot as plt

plt.plot(model.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

## 7. Tips for Effective Training

- **Start** with a small learning rate; increase if loss stagnates.
- **Use** mini-batches for large datasets.
- **Monitor** both loss and accuracy.
- **Shuffle** data to prevent overfitting.

## Next Steps

- **Load real datasets** → See `dataset_loading.md`
- **Build more complex networks** → See `basic_network.md`
- **Check API reference** → Explore layer classes and functions


© 2025 VisuaLife | Written by Jeewan Aidi
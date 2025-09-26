# tests/test_model_save_load.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualife.core.model import Model
from visualife.core.layers import Dense
from visualife.core.convolutional import Conv2D, MaxPool2D, Flatten
from visualife.core.activations import ReLU

# --- Create a small dummy CNN model ---
model = Model()

# Small ConvNet for testing
model.add(Conv2D(num_filters=8, filter_size=3, stride=1, padding=1))  # small for test
model.add(ReLU())
model.add(MaxPool2D(pool_size=2, stride=2))

model.add(Conv2D(num_filters=16, filter_size=3, stride=1, padding=1))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2, stride=2))

model.add(Flatten())

model.add(Dense(16*4*4, 32))  # small dense layer
model.add(ReLU())
model.add(Dense(32, 10))      # output layer for 10 classes

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', learning_rate=0.01)

# Dummy data (batch of 8, 16x16 RGB images)
X_dummy = np.random.rand(8, 16, 16, 3)
y_dummy = np.eye(10)[np.random.randint(0, 10, size=8)]  # one-hot labels

# Train a few steps
model.fit(X_dummy, y_dummy, epochs=2, batch_size=4, verbose=0)

# Save model
save_path = "test_weights.pkl"
model.save(save_path)

# Change weights manually to check reload
for layer in model.layers:
    if hasattr(layer, 'weights'):
        layer.weights += np.random.rand(*layer.weights.shape)

# Load model back
model.load(save_path)

# Check if weights match original after load
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'weights'):
        print(f"Layer {i} weights after load:\n", layer.weights)

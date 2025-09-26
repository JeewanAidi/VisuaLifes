# experiments/train_real_dataset.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from visualife.utils.data_loader import DataLoader
from visualife.core.model import Model
from visualife.core.convolutional import Conv2D, MaxPool2D, Flatten
from visualife.core.layers import Dense
from visualife.core.activations import ReLU, Softmax

# =======================
# CONFIG
# =======================
train_dir = "dataset_resized/train"
test_dir = "dataset_resized/test"
img_size = (128, 128)
num_classes = 9
batch_size = 32
epochs = 1
learning_rate = 0.01

# =======================
# LOAD DATASET
# =======================
print("📂 Loading dataset...")

train_loader = DataLoader(
    data_dir=train_dir,
    img_size=img_size,
    batch_size=batch_size,
    num_classes=num_classes,
    shuffle=True
)

test_loader = DataLoader(
    data_dir=test_dir,
    img_size=img_size,
    batch_size=batch_size,
    num_classes=num_classes,
    shuffle=False
)

print("✅ Dataset loaded")

# =======================
# BUILD MODEL (lighter CNN)
# =======================
print("🧠 Building model...")

model = Model()

model.add(Conv2D(8, 3, stride=1, padding=1))   # 8 filters only
model.add(ReLU())
model.add(MaxPool2D(pool_size=2, stride=2))    # -> 64x64x8

model.add(Conv2D(16, 3, stride=1, padding=1))  # 16 filters
model.add(ReLU())
model.add(MaxPool2D(pool_size=2, stride=2))    # -> 32x32x16

model.add(Conv2D(32, 3, stride=1, padding=1))  # 32 filters
model.add(ReLU())
model.add(MaxPool2D(pool_size=2, stride=2))    # -> 16x16x32

model.add(Flatten())                           # -> 8192 features
model.add(Dense(8192, 64))                     # small dense
model.add(ReLU())
model.add(Dense(64, num_classes))              # final classifier
model.add(Softmax())


# Compile
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    learning_rate=learning_rate
)

print("✅ Model built & compiled")

# =======================
# TRAIN
# =======================
print("🚀 Starting training...")

for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        loss, acc = model.train_step(X_batch, y_batch)
        epoch_loss += loss
        epoch_acc += acc
        num_batches += 1

    epoch_loss /= num_batches
    epoch_acc /= num_batches

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

print("🎉 Training finished!")

# =======================
# EVALUATE
# =======================
print("🔍 Evaluating on test set...")
test_loss = 0
test_acc = 0
num_batches = 0

from visualife.core.losses import CrossEntropyLoss
from visualife.core.metrics import Accuracy

loss_fn = CrossEntropyLoss()
acc_fn = Accuracy()

for X_batch, y_batch in test_loader:
    y_pred = model.forward(X_batch, training=False)

    loss = loss_fn.forward(y_pred, y_batch)
    acc = acc_fn.forward(y_pred, y_batch)

    test_loss += loss
    test_acc += acc
    num_batches += 1

test_loss /= num_batches
test_acc /= num_batches

print(f"✅ Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# =======================
# SAVE MODEL
# =======================
model.save("experiments/real_dataset_cnn.pkl")
print("💾 Model saved at experiments/real_dataset_cnn_light.pkl")

# test_metrics.py
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualife.core.metrics import Accuracy

def test_accuracy_metric():
    print("🧪 Testing Accuracy Metric...")

    metric = Accuracy()

    # Test 1: Perfect predictions
    y_true = np.array([[1,0,0],[0,1,0],[0,0,1]])  # One-hot
    y_pred = np.array([[1,0,0],[0,1,0],[0,0,1]])
    acc = metric.forward(y_pred, y_true)
    assert np.isclose(acc, 1.0), f"Expected 1.0, got {acc}"
    print("✅ Perfect prediction test passed!")

    # Test 2: Half correct
    y_pred = np.array([[1,0,0],[1,0,0],[0,0,1]])
    acc = metric.forward(y_pred, y_true)
    assert np.isclose(acc, 2/3), f"Expected 0.6667, got {acc}"
    print("✅ Half correct prediction test passed!")

    # Test 3: All incorrect
    y_pred = np.array([[0,1,0],[0,0,1],[1,0,0]])
    acc = metric.forward(y_pred, y_true)
    assert np.isclose(acc, 0.0), f"Expected 0.0, got {acc}"
    print("✅ All incorrect prediction test passed!")

    # Test 4: Random prediction
    y_pred = np.array([[0.7,0.2,0.1],[0.1,0.3,0.6],[0.2,0.5,0.3]])
    acc = metric.forward(y_pred, y_true)
    expected = 1/3  # Only the first one is correct
    assert np.isclose(acc, expected), f"Expected {expected}, got {acc}"
    print("✅ Random prediction test passed!")

# -----------------------------
# Run test
# -----------------------------
if __name__ == "__main__":
    test_accuracy_metric()
    print("🎉 All Accuracy metric tests passed!")

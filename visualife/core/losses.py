import numpy as np

class MeanSquaredError:
    """
    Mean Squared Error loss for regression tasks.
    forward: L = 1/n * Σ(y_true - y_pred)²
    backward: dL/dy_pred = -2/n * (y_true - y_pred)
    """
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self):
        batch_size = self.y_true.shape[0]
        return -2 * (self.y_true - self.y_pred) / batch_size


class BinaryCrossEntropy:
    """
    Binary Cross Entropy loss for binary classification.
    """
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.batch_size = None
    
    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_true.shape[0]
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self):
        # Gradient: (y_pred - y_true) / (batch_size * y_pred * (1 - y_pred))
        grad = (self.y_pred - self.y_true) / (self.batch_size * self.y_pred * (1 - self.y_pred) + 1e-15)
        grad = np.clip(grad, -1e10, 1e10)  # extra stability
        return grad


class CrossEntropyLoss:
    """
    Categorical Cross Entropy loss for multi-class classification.
    Supports both one-hot and sparse integer labels.
    """
    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = y_true.shape[0]

        if y_true.ndim == 1:  # sparse labels
            loss = -np.mean(np.log(y_pred[np.arange(batch_size), y_true]))
        else:  # one-hot labels
            loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        return loss
    
    def backward(self):
        batch_size = self.y_true.shape[0]
        if self.y_true.ndim == 1:  # sparse
            grad = self.y_pred.copy()
            grad[np.arange(batch_size), self.y_true] -= 1
            grad /= batch_size
        else:  # one-hot
            grad = (self.y_pred - self.y_true) / batch_size
        return grad
# metrics.py
import numpy as np

class Accuracy:
    """
    Accuracy metric for multi-class classification.
    Works with one-hot encoded targets and predicted probabilities.
    """
    def __init__(self):
        self.name = "accuracy"

    def forward(self, y_pred, y_true):
        """
        Compute accuracy for a batch.

        Parameters:
        - y_pred: np.array of shape (batch_size, num_classes), predicted probabilities
        - y_true: np.array of shape (batch_size, num_classes), one-hot encoded labels

        Returns:
        - accuracy: float, fraction of correct predictions
        """
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy

    def reset(self):
        """Optional: reset internal state if you accumulate metrics over multiple batches"""
        pass

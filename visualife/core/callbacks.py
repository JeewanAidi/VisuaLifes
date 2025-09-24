import copy
import numpy as np

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.
    Keeps best weights and can restore them.
    """
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_loss = np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, loss, model):
        improved = loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = loss
            self.wait = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = self._get_model_weights(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None and model is not None:
                    self._set_model_weights(model, self.best_weights)
                return True  # Stop training
        return False  # Continue training

    def _get_model_weights(self, model):
        weights = []
        for layer in model.layers:
            layer_weights = {}
            if hasattr(layer, 'weights') and layer.weights is not None:
                layer_weights['weights'] = layer.weights.copy()
            if hasattr(layer, 'biases') and layer.biases is not None:
                layer_weights['biases'] = layer.biases.copy()
            weights.append(layer_weights)
        return weights

    def _set_model_weights(self, model, weights):
        for layer, layer_weights in zip(model.layers, weights):
            if 'weights' in layer_weights and layer_weights['weights'] is not None:
                layer.weights = layer_weights['weights'].copy()
            if 'biases' in layer_weights and layer_weights['biases'] is not None:
                layer.biases = layer_weights['biases'].copy()


class LearningRateScheduler:
    """
    Learning rate scheduler that reduces LR on plateau.
    """
    def __init__(self, factor=0.5, patience=3, min_lr=1e-6, min_delta=1e-4):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, loss, model, optimizer):
        improved = loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(optimizer.learning_rate * self.factor, self.min_lr)
                if new_lr != optimizer.learning_rate:
                    print(f"Reducing learning rate from {optimizer.learning_rate:.6f} to {new_lr:.6f}")
                    optimizer.learning_rate = new_lr
                    self.wait = 0

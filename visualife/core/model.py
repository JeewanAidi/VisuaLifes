import numpy as np
import pickle
from visualife.core.losses import CrossEntropyLoss, MeanSquaredError, BinaryCrossEntropy
from visualife.core.optimizers import SGD, Momentum, Adam
from visualife.core.callbacks import EarlyStopping, LearningRateScheduler
from visualife.core.layers import Dropout

class Model:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.history = {'train_loss': [], 'train_accuracy': [],
                        'val_loss': [], 'val_accuracy': [],
                        'learning_rate': []}
        self.current_epoch = 0

    def add(self, layer):
        self.layers.append(layer)
        print(f"Added {layer.__class__.__name__} layer")

    def compile(self, loss='categorical_crossentropy',
                optimizer='adam', learning_rate=0.001, **optimizer_params):
        if loss == 'categorical_crossentropy':
            self.loss_function = CrossEntropyLoss()
        elif loss == 'mean_squared_error':
            self.loss_function = MeanSquaredError()
        elif loss == 'binary_crossentropy':
            self.loss_function = BinaryCrossEntropy()
        else:
            raise ValueError(f"Unsupported loss: {loss}")

        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate, **optimizer_params)
        elif optimizer == 'momentum':
            self.optimizer = Momentum(learning_rate, **optimizer_params)
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        print(f"Model compiled with {loss}, optimizer={optimizer}, lr={learning_rate}")

    def forward(self, X, training=True):
        out = X
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            out = layer.forward(out)
        return out

    def backward(self, dZ):
        grad = dZ
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                # Pass learning_rate only if layer has trainable parameters
                if hasattr(layer, 'weights') or hasattr(layer, 'gamma'):
                    grad = layer.backward(grad, self.optimizer.learning_rate)
                else:
                    grad = layer.backward(grad)

        # Batch update of all trainable parameters in one vectorized pass
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'dweights') and layer.dweights is not None:
                self.optimizer.update(layer)

        return grad

    def compute_loss(self, y_pred, y_true):
        # Forward loss
        loss = self.loss_function.forward(y_pred, y_true)
        # Regularization loss
        reg_loss = sum(layer.regularization_loss() for layer in self.layers if hasattr(layer, 'regularization_loss'))
        loss += reg_loss
        # Vectorized backward
        dZ = self.loss_function.backward()
        return loss, dZ

    def train_step(self, X_batch, y_batch):
        y_pred = self.forward(X_batch, training=True)
        loss, dZ = self.compute_loss(y_pred, y_batch)
        self.backward(dZ)

        # Accuracy vectorized
        if y_batch.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
        else:
            predictions = (y_pred > 0.5).astype(int).flatten()
            true_labels = y_batch.flatten()
        accuracy = np.mean(predictions == true_labels)
        return loss, accuracy

    def fit(self, X_train, y_train, epochs=10, batch_size=32,
            validation_data=None, callbacks=None, verbose=1):
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        callbacks = callbacks or []

        early_stopper = next((c for c in callbacks if isinstance(c, EarlyStopping)), None)
        lr_scheduler = next((c for c in callbacks if isinstance(c, LearningRateScheduler)), None)

        self.history = {k: [] for k in self.history}

        for epoch in range(epochs):
            self.current_epoch = epoch
            idx = np.random.permutation(n_samples)
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]

            epoch_loss = 0
            epoch_acc = 0

            for batch in range(n_batches):
                start, end = batch * batch_size, min((batch+1)*batch_size, n_samples)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                batch_loss, batch_acc = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
                epoch_acc += batch_acc

            epoch_loss /= n_batches
            epoch_acc /= n_batches

            self.history['train_loss'].append(epoch_loss)
            self.history['train_accuracy'].append(epoch_acc)
            self.history['learning_rate'].append(self.optimizer.learning_rate)

            # Validation
            if validation_data:
                X_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(X_val, y_val, verbose=0)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
            else:
                val_loss, val_acc = None, None

            if verbose:
                val_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if validation_data else ""
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}{val_str}, LR: {self.optimizer.learning_rate:.6f}")

            # Callbacks
            if early_stopper and val_loss is not None and early_stopper.on_epoch_end(epoch, val_loss, self):
                print(f"Early stopping at epoch {epoch+1}")
                break
            if lr_scheduler and val_loss is not None:
                lr_scheduler.on_epoch_end(epoch, val_loss, self, self.optimizer)

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict
        return History(self.history)

    def evaluate(self, X, y, verbose=1):
        y_pred = self.forward(X, training=False)
        loss = self.loss_function.forward(y_pred, y)
        reg_loss = sum(layer.regularization_loss() for layer in self.layers if hasattr(layer, 'regularization_loss'))
        loss += reg_loss

        if y.shape[1] > 1:
            pred = np.argmax(y_pred, axis=1)
            true = np.argmax(y, axis=1)
        else:
            pred = (y_pred > 0.5).astype(int).flatten()
            true = y.flatten()
        acc = np.mean(pred == true)

        if verbose:
            print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return loss, acc

    def predict(self, X):
        return self.forward(X, training=False)

    def predict_classes(self, X):
        y_pred = self.predict(X)
        if y_pred.shape[1] > 1:
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred > 0.5).astype(int)

    # --- Save & Load ---
    def save(self, filepath):
        model_data = {
            'layers': [],
            'optimizer': {'learning_rate': self.optimizer.learning_rate},
            'history': self.history
        }
        for layer in self.layers:
            layer_dict = {}
            for attr in ['weights', 'biases', 'gamma', 'beta', 'running_mean', 'running_var']:
                if hasattr(layer, attr):
                    layer_dict[attr] = getattr(layer, attr).copy()
            model_data['layers'].append(layer_dict)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        for layer, layer_dict in zip(self.layers, model_data['layers']):
            for attr, val in layer_dict.items():
                setattr(layer, attr, val.copy())
        if 'optimizer' in model_data:
            self.optimizer.learning_rate = model_data['optimizer']['learning_rate']
        if 'history' in model_data:
            self.history = model_data['history']

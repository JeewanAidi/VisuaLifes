import numpy as np
import pickle
import sys
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

    def initialize_weights(self, input_shape):
        """
        Initialize all layer weights using He initialization before training starts
        """
        print("🔧 Initializing model weights with He initialization...")
        
        # Run a dummy forward pass to trigger weight initialization
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        current_output = dummy_input
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'initialize_parameters'):
                # For layers that need explicit input shape
                if hasattr(layer, 'input_size'):  # Dense layer
                    layer.initialize_parameters()
                else:  # Conv2D layer
                    if hasattr(current_output, 'shape'):
                        input_channels = current_output.shape[-1]
                        layer.initialize_parameters(input_channels)
            
            # Forward pass to get output shape for next layer
            if hasattr(layer, 'forward'):
                current_output = layer.forward(current_output)
        
        # Verify weights are initialized
        total_params = self.count_parameters()
        print(f"✅ Model weights initialized - {total_params} parameters")
        return total_params

    def count_parameters(self):
        """Count total trainable parameters"""
        total_params = 0
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'filters') and layer.filters is not None:
                total_params += np.prod(layer.filters.shape)
            if hasattr(layer, 'bias') and layer.bias is not None:
                total_params += np.prod(layer.bias.shape)
            if hasattr(layer, 'weights') and layer.weights is not None:
                total_params += np.prod(layer.weights.shape)
            if hasattr(layer, 'biases') and layer.biases is not None:
                total_params += np.prod(layer.biases.shape)
            if hasattr(layer, 'gamma') and layer.gamma is not None:
                total_params += np.prod(layer.gamma.shape)
            if hasattr(layer, 'beta') and layer.beta is not None:
                total_params += np.prod(layer.beta.shape)
        return total_params

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
                if hasattr(layer, 'weights') or hasattr(layer, 'gamma'):
                    grad = layer.backward(grad, self.optimizer.learning_rate)
                else:
                    grad = layer.backward(grad)
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'dweights') and layer.dweights is not None:
                self.optimizer.update(layer)
        return grad

    def compute_loss(self, y_pred, y_true):
        loss = self.loss_function.forward(y_pred, y_true)
        reg_loss = sum(layer.regularization_loss() for layer in self.layers if hasattr(layer, 'regularization_loss'))
        loss += reg_loss
        dZ = self.loss_function.backward()
        return loss, dZ

    def train_step(self, X_batch, y_batch):
        y_pred = self.forward(X_batch, training=True)
        loss, dZ = self.compute_loss(y_pred, y_batch)
        self.backward(dZ)
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
        # Initialize weights if not already done
        if self.count_parameters() == 0:
            self.initialize_weights(X_train.shape[1:])
            
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

            if early_stopper and val_loss is not None and early_stopper.on_epoch_end(epoch, val_loss, self):
                print(f"Early stopping at epoch {epoch+1}")
                break
            if lr_scheduler and val_loss is not None:
                lr_scheduler.on_epoch_end(epoch, val_loss, self, self.optimizer)

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

    def save(self, filepath):
        """
        Save full model: architecture, weights, optimizer, history.
        """
        model_data = {
            'layers': [],
            'optimizer': None,
            'history': self.history
        }

        for layer in self.layers:
            layer_dict = {
                'class_name': layer.__class__.__name__,
                'config': layer.get_config() if hasattr(layer, 'get_config') else {},
                'weights': {}
            }
            
            # Handle different layer types and their weight attribute names
            weight_attrs = []
            if hasattr(layer, 'filters'):  # Conv2D
                weight_attrs = ['filters', 'bias']
            elif hasattr(layer, 'weights'):  # Dense
                weight_attrs = ['weights', 'biases']
            elif hasattr(layer, 'gamma'):  # BatchNorm
                weight_attrs = ['gamma', 'beta', 'running_mean', 'running_var']
            else:  # Activation layers, etc.
                weight_attrs = []
            
            for attr in weight_attrs:
                if hasattr(layer, attr) and getattr(layer, attr) is not None:
                    layer_dict['weights'][attr] = getattr(layer, attr).copy()
            
            model_data['layers'].append(layer_dict)

        if self.optimizer is not None:
            model_data['optimizer'] = {
                'class_name': self.optimizer.__class__.__name__,
                'learning_rate': self.optimizer.learning_rate
            }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model fully saved to {filepath}")

    def load(self, filepath):
        """
        Load full model: reconstruct layers from saved architecture + load weights + optimizer.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.layers = []
        for layer_dict in model_data['layers']:
            class_name = layer_dict['class_name']
            config = layer_dict['config']

            # Reconstruct layer
            LayerClass = getattr(sys.modules['visualife.core.layers'], class_name, None) \
                        or getattr(sys.modules['visualife.core.convolutional'], class_name, None) \
                        or getattr(sys.modules['visualife.core.activations'], class_name, None)
            if LayerClass is None:
                raise ValueError(f"Unknown layer class: {class_name}")

            layer = LayerClass(**config) if config else LayerClass()

            # Load weights - handle different attribute names
            for attr, val in layer_dict['weights'].items():
                if hasattr(layer, attr):
                    setattr(layer, attr, val.copy())
                else:
                    print(f"⚠️ Warning: Layer {class_name} has no attribute '{attr}'")

            self.layers.append(layer)

        # Load optimizer
        opt_data = model_data.get('optimizer')
        if opt_data:
            OptimizerClass = getattr(sys.modules['visualife.core.optimizers'], opt_data['class_name'])
            self.optimizer = OptimizerClass(opt_data['learning_rate'])

        # Load history
        self.history = model_data.get('history', {})
        print(f"✅ Model fully loaded from {filepath}")
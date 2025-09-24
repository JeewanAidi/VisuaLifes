import numpy as np

class ReLU:
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    
    def backward(self, dZ, learning_rate=None):
        return dZ * (self.input > 0).astype(dZ.dtype)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, X):
        self.input = X
        return np.where(X > 0, X, self.alpha * X)
    
    def backward(self, dZ, learning_rate=None):
        return dZ * np.where(self.input > 0, 1, self.alpha)


class ParametricReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, X):
        self.input = X
        return np.where(X > 0, X, self.alpha * X)
    
    def backward(self, dZ, learning_rate=None):
        return dZ * np.where(self.input > 0, 1, self.alpha)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, X):
        self.output = np.where(X > 0, X, self.alpha * (np.exp(X) - 1))
        return self.output
    
    def backward(self, dZ, learning_rate=None):
        grad = np.where(self.output > 0, 1, self.output + self.alpha)
        return dZ * grad


class Swish:
    def forward(self, X):
        X_clipped = np.clip(X, -500, 500)
        self.sigmoid = 1 / (1 + np.exp(-X_clipped))
        self.output = X * self.sigmoid
        return self.output
    
    def backward(self, dZ, learning_rate=None):
        grad = self.sigmoid + self.output * (1 - self.sigmoid)
        return dZ * grad


class Sigmoid:
    def forward(self, X):
        X_clipped = np.clip(X, -500, 500)
        self.output = 1 / (1 + np.exp(-X_clipped))
        return self.output
    
    def backward(self, dZ, learning_rate=None):
        return dZ * self.output * (1 - self.output)


class Softmax:
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dZ, learning_rate=None):
        return dZ  # handled by CrossEntropyLoss


class Tanh:
    def forward(self, X):
        self.output = np.tanh(X)
        return self.output
    
    def backward(self, dZ, learning_rate=None):
        return dZ * (1 - self.output ** 2)

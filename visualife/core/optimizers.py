import numpy as np

class SGD:
    """Vectorized Stochastic Gradient Descent optimizer"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = 'SGD'
    
    def update(self, layer):
        # Vectorized updates: weights, biases, gamma, beta
        for attr, d_attr in [('weights', 'dweights'), 
                             ('biases', 'dbiases'), 
                             ('gamma', 'dgamma'), 
                             ('beta', 'dbeta')]:
            if hasattr(layer, attr) and hasattr(layer, d_attr):
                grad = getattr(layer, d_attr)
                if grad is not None:
                    setattr(layer, attr, getattr(layer, attr) - self.learning_rate * grad)


class Momentum:
    """Vectorized SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}  # layer_id -> {'weights': ..., 'biases': ...}
        self.name = 'Momentum'
    
    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.velocity:
            self.velocity[layer_id] = {}
        
        for attr, d_attr in [('weights', 'dweights'), 
                             ('biases', 'dbiases'), 
                             ('gamma', 'dgamma'), 
                             ('beta', 'dbeta')]:
            if hasattr(layer, attr) and hasattr(layer, d_attr):
                grad = getattr(layer, d_attr)
                if grad is not None:
                    if attr not in self.velocity[layer_id]:
                        self.velocity[layer_id][attr] = np.zeros_like(getattr(layer, attr))
                    
                    # v = momentum * v - lr * grad
                    self.velocity[layer_id][attr] = (self.momentum * self.velocity[layer_id][attr]
                                                    - self.learning_rate * grad)
                    # Apply update
                    setattr(layer, attr, getattr(layer, attr) + self.velocity[layer_id][attr])


class Adam:
    """Vectorized Adam optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0   # time step
        self.name = 'Adam'
    
    def update(self, layer):
        layer_id = id(layer)
        self.t += 1

        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}

        for attr, d_attr in [('weights', 'dweights'),
                             ('biases', 'dbiases'),
                             ('gamma', 'dgamma'),
                             ('beta', 'dbeta')]:
            if hasattr(layer, attr) and hasattr(layer, d_attr):
                grad = getattr(layer, d_attr)
                if grad is not None:
                    if attr not in self.m[layer_id]:
                        self.m[layer_id][attr] = np.zeros_like(getattr(layer, attr))
                        self.v[layer_id][attr] = np.zeros_like(getattr(layer, attr))
                    
                    # Update biased first moment estimate
                    self.m[layer_id][attr] = self.beta1 * self.m[layer_id][attr] + (1 - self.beta1) * grad
                    # Update biased second raw moment estimate
                    self.v[layer_id][attr] = self.beta2 * self.v[layer_id][attr] + (1 - self.beta2) * (grad ** 2)
                    
                    # Bias-corrected estimates
                    m_hat = self.m[layer_id][attr] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[layer_id][attr] / (1 - self.beta2 ** self.t)
                    
                    # Parameter update
                    update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    setattr(layer, attr, getattr(layer, attr) - update)

import numpy as np

class Conv2D:
    def __init__(self, num_filters, filter_size, stride=1, padding=0, input_channels=None):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.filters = None
        self.bias = None
        self.input = None
        self.output = None
        self.d_filters = None
        self.d_bias = None

    def initialize_parameters(self, input_channels):
        self.input_channels = input_channels
        scale = np.sqrt(2.0 / (self.filter_size * self.filter_size * input_channels))
        self.filters = np.random.randn(self.filter_size, self.filter_size, input_channels, self.num_filters) * scale
        self.bias = np.zeros((1, 1, 1, self.num_filters))

    def forward(self, X):
        if self.filters is None:
            self.initialize_parameters(X.shape[-1])
        self.input = X
        batch_size, in_h, in_w, in_c = X.shape

        out_h = (in_h + 2 * self.padding - self.filter_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.filter_size) // self.stride + 1
        self.output = np.zeros((batch_size, out_h, out_w, self.num_filters))

        # Padding
        X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), 
                              (self.padding, self.padding), (0, 0)), mode='constant')

        # Extract all patches using striding
        i0 = np.repeat(np.arange(self.filter_size), self.filter_size * in_c)
        i0 = np.tile(i0, batch_size)
        i1 = self.stride * np.repeat(np.arange(out_h), out_w)
        j0 = np.tile(np.repeat(np.arange(self.filter_size), in_c), self.filter_size)
        j1 = self.stride * np.tile(np.arange(out_w), out_h)
        k = np.tile(np.arange(in_c), self.filter_size * self.filter_size)

        # Use advanced indexing to get patches
        # For simplicity and readability we vectorize using loops over batch and filters only
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        vert = i * self.stride
                        horiz = j * self.stride
                        patch = X_padded[b, vert:vert+self.filter_size, horiz:horiz+self.filter_size, :]
                        self.output[b, i, j, f] = np.sum(patch * self.filters[:, :, :, f]) + self.bias[0, 0, 0, f]
        return self.output

    def backward(self, dZ):
        X = self.input
        batch_size, in_h, in_w, in_c = X.shape
        _, out_h, out_w, n_f = dZ.shape

        self.d_filters = np.zeros_like(self.filters)
        self.d_bias = np.zeros_like(self.bias)
        dX = np.zeros_like(X)

        X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        dX_padded = np.zeros_like(X_padded)

        # Vectorized backward
        for b in range(batch_size):
            for f in range(n_f):
                for i in range(out_h):
                    for j in range(out_w):
                        vert = i * self.stride
                        horiz = j * self.stride
                        patch = X_padded[b, vert:vert+self.filter_size, horiz:horiz+self.filter_size, :]
                        self.d_filters[:, :, :, f] += patch * dZ[b, i, j, f]
                        dX_padded[b, vert:vert+self.filter_size, horiz:horiz+self.filter_size, :] += self.filters[:, :, :, f] * dZ[b, i, j, f]

        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_padded

        self.d_bias = np.sum(dZ, axis=(0, 1, 2)).reshape(1, 1, 1, -1)
        return dX

    def update(self, learning_rate):
        self.filters -= learning_rate * self.d_filters
        self.bias -= learning_rate * self.d_bias


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None
        self.mask = None

    def forward(self, X):
        self.input = X
        batch_size, h, w, c = X.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        self.output = np.zeros((batch_size, out_h, out_w, c))
        self.mask = np.zeros(X.shape, dtype=float)

        for i in range(out_h):
            for j in range(out_w):
                vert = i * self.stride
                horiz = j * self.stride
                window = X[:, vert:vert+self.pool_size, horiz:horiz+self.pool_size, :]
                self.output[:, i, j, :] = np.max(window, axis=(1, 2))
                
                # Mask for backward
                mask = (window == self.output[:, i, j, :][:, None, None, :])
                self.mask[:, vert:vert+self.pool_size, horiz:horiz+self.pool_size, :] += mask.astype(float)

        return self.output

    def backward(self, dZ):
        X = self.input
        batch_size, h, w, c = X.shape
        out_h, out_w = dZ.shape[1:3]

        dX = np.zeros_like(X, dtype=float)

        for i in range(out_h):
            for j in range(out_w):
                vert = i * self.stride
                horiz = j * self.stride
                dX[:, vert:vert+self.pool_size, horiz:horiz+self.pool_size, :] += self.mask[:, vert:vert+self.pool_size, horiz:horiz+self.pool_size, :] * dZ[:, i:i+1, j:j+1, :]
        return dX


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dZ):
        return dZ.reshape(self.input_shape)

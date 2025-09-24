import numpy as np

# ============================================================
# Utility functions (im2col & col2im for convolution/pooling)
# ============================================================
def get_im2col_indices(X_shape, field_height, field_width, padding, stride):
    N, H, W, C = X_shape
    out_h = (H + 2 * padding - field_height) // stride + 1
    out_w = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (i, j, k)


def im2col(X, field_height, field_width, padding, stride):
    N, H, W, C = X.shape
    X_padded = np.pad(X, ((0,0),(padding,padding),(padding,padding),(0,0)), mode="constant")
    i, j, k = get_im2col_indices(X.shape, field_height, field_width, padding, stride)
    cols = X_padded[:, i, j, k]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im(cols, X_shape, field_height, field_width, padding, stride):
    N, H, W, C = X_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    X_padded = np.zeros((N, H_padded, W_padded, C), dtype=cols.dtype)

    i, j, k = get_im2col_indices(X_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(field_height * field_width * C, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(X_padded, (slice(None), i, j, k), cols_reshaped)

    if padding == 0:
        return X_padded
    return X_padded[:, padding:-padding, padding:-padding, :]


# ============================================================
# Conv2D Layer (vectorized with im2col)
# ============================================================
class Conv2D:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = None
        self.bias = None
        self.d_filters = None
        self.d_bias = None

    def initialize_parameters(self, input_channels):
        scale = np.sqrt(2.0 / (self.filter_size * self.filter_size * input_channels))
        self.filters = np.random.randn(
            self.num_filters, input_channels * self.filter_size * self.filter_size
        ) * scale
        self.bias = np.zeros((self.num_filters, 1))

    def forward(self, X):
        if self.filters is None:
            self.initialize_parameters(X.shape[-1])
        self.X_shape = X.shape
        N, H, W, C = X.shape

        self.X_col = im2col(X, self.filter_size, self.filter_size, self.padding, self.stride)
        W_col = self.filters.reshape(self.num_filters, -1)  # ✅ reshape
        out = W_col @ self.X_col + self.bias

        out_h = (H + 2*self.padding - self.filter_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.filter_size) // self.stride + 1
        out = out.reshape(self.num_filters, out_h, out_w, N)
        out = out.transpose(3, 1, 2, 0)  # (N, out_h, out_w, num_filters)
        self.out = out
        return out

    def backward(self, d_out, learning_rate=0.001):
        N, H, W, C = self.X_shape
        d_out_reshaped = d_out.transpose(3, 1, 2, 0).reshape(self.num_filters, -1)

        dW = d_out_reshaped @ self.X_col.T
        self.d_filters = dW.reshape(self.filters.shape)  # ✅ store gradients
        self.d_bias = np.sum(d_out_reshaped, axis=1, keepdims=True)

        W_col = self.filters.reshape(self.num_filters, -1)
        dX_col = W_col.T @ d_out_reshaped
        dX = col2im(dX_col, self.X_shape, self.filter_size, self.filter_size, self.padding, self.stride)

        # inline update
        self.filters -= learning_rate * self.d_filters
        self.bias -= learning_rate * self.d_bias
        return dX

    def update(self, learning_rate=0.001):  # ✅ new method
        if self.d_filters is not None and self.d_bias is not None:
            self.filters -= learning_rate * self.d_filters
            self.bias -= learning_rate * self.d_bias



# ============================================================
# MaxPool2D Layer (vectorized with im2col)
# ============================================================
class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X_shape = X.shape
        N, H, W, C = X.shape

        self.X_col = im2col(X, self.pool_size, self.pool_size, 0, self.stride)
        self.X_col = self.X_col.reshape(self.pool_size * self.pool_size, -1)

        self.max_idx = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_idx, np.arange(self.max_idx.size)]

        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        out = out.reshape(out_h, out_w, C, N).transpose(3, 0, 1, 2)
        return out

    def backward(self, d_out):
        d_out_flat = d_out.transpose(1, 2, 3, 0).ravel()
        dX_col = np.zeros_like(self.X_col)
        dX_col[self.max_idx, np.arange(self.max_idx.size)] = d_out_flat
        dX = col2im(dX_col, self.X_shape, self.pool_size, self.pool_size, 0, self.stride)
        return dX


# ============================================================
# Flatten Layer
# ============================================================
class Flatten:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)

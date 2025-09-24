# tests/test_layers.py
import numpy as np
from visualife.core.layers import Dense, Dropout, BatchNorm


def test_dense_forward_backward():
    np.random.seed(42)
    X = np.random.randn(5, 4)  # batch=5, features=4
    dense = Dense(4, 3, l1_reg=0.01, l2_reg=0.01)
    out = dense.forward(X)

    assert out.shape == (5, 3), "Dense forward output shape mismatch"

    dZ = np.random.randn(5, 3)
    dX = dense.backward(dZ)

    assert dX.shape == (5, 4), "Dense backward output shape mismatch"
    print("✅ Dense forward/backward passed")


def test_dropout_forward_backward():
    np.random.seed(42)
    X = np.ones((4, 4))
    dropout = Dropout(rate=0.5)
    dropout.training = True

    out = dropout.forward(X)
    assert out.shape == X.shape, "Dropout forward shape mismatch"

    dZ = np.ones((4, 4))
    dX = dropout.backward(dZ)
    assert dX.shape == X.shape, "Dropout backward shape mismatch"
    print("✅ Dropout forward/backward passed")


def test_batchnorm_forward_backward():
    np.random.seed(42)
    X = np.random.randn(6, 4)
    bn = BatchNorm(4)
    bn.training = True

    out = bn.forward(X)
    assert out.shape == (6, 4), "BatchNorm forward shape mismatch"

    dZ = np.random.randn(6, 4)
    dX = bn.backward(dZ)
    assert dX.shape == (6, 4), "BatchNorm backward shape mismatch"
    print("✅ BatchNorm forward/backward passed")


if __name__ == "__main__":
    test_dense_forward_backward()
    test_dropout_forward_backward()
    test_batchnorm_forward_backward()
    print("\n🎉 All layer tests finished successfully")

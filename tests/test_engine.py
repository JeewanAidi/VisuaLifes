# tests/test_ops.py
import numpy as np
from visualife.core.engine import matrix_multiply, elementwise_multiply, sum_array

def test_matrix_multiply():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    expected = A @ B
    result = matrix_multiply(A, B)
    assert np.allclose(result, expected)
    print("✅ test_matrix_multiply passed")

def test_matrix_multiply_invalid_shape():
    try:
        A = np.array([[1, 2, 3]])
        B = np.array([[1, 2]])
        matrix_multiply(A, B)
    except ValueError:
        print("✅ test_matrix_multiply_invalid_shape passed")
    else:
        print("❌ test_matrix_multiply_invalid_shape failed")

def test_elementwise_multiply():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 0], [1, 5]])
    expected = A * B
    result = elementwise_multiply(A, B)
    assert np.allclose(result, expected)
    print("✅ test_elementwise_multiply passed")

def test_elementwise_multiply_invalid_shape():
    try:
        A = np.array([[1, 2]])
        B = np.array([[1, 2], [3, 4]])
        elementwise_multiply(A, B)
    except ValueError:
        print("✅ test_elementwise_multiply_invalid_shape passed")
    else:
        print("❌ test_elementwise_multiply_invalid_shape failed")

def test_sum_array_none():
    A = np.array([[1, 2], [3, 4]])
    expected = np.sum(A)
    result = sum_array(A, axis=None)
    assert np.allclose(result, expected)
    print("✅ test_sum_array_none passed")

def test_sum_array_axis0():
    A = np.array([[1, 2], [3, 4]])
    expected = np.sum(A, axis=0, keepdims=True)
    result = sum_array(A, axis=0)
    assert np.allclose(result, expected)
    print("✅ test_sum_array_axis0 passed")

def test_sum_array_axis1():
    A = np.array([[1, 2], [3, 4]])
    expected = np.sum(A, axis=1, keepdims=True)
    result = sum_array(A, axis=1)
    assert np.allclose(result, expected)
    print("✅ test_sum_array_axis1 passed")

def test_sum_array_invalid_axis():
    A = np.array([[1, 2], [3, 4]])
    try:
        sum_array(A, axis=2)
    except ValueError:
        print("✅ test_sum_array_invalid_axis passed")
    else:
        print("❌ test_sum_array_invalid_axis failed")

if __name__ == "__main__":
    test_matrix_multiply()
    test_matrix_multiply_invalid_shape()
    test_elementwise_multiply()
    test_elementwise_multiply_invalid_shape()
    test_sum_array_none()
    test_sum_array_axis0()
    test_sum_array_axis1()
    test_sum_array_invalid_axis()
    print("\n🎉 All tests finished")

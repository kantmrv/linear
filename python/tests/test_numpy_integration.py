"""
Test suite for NumPy integration and interoperability
"""

import numpy as np

from linear import Matrix


def test_numpy_roundtrip():
    """Test NumPy → Matrix → NumPy conversion"""
    arr_in = np.array([[1, 2], [3, 4]], dtype=np.float64)
    m = Matrix(arr_in)
    arr_out = np.array(m)

    assert np.allclose(arr_in, arr_out)


def test_matrix_from_numpy():
    """Test creating Matrix from NumPy array"""
    arr = np.random.rand(3, 4)
    m = Matrix(arr)

    assert m.rows == 3 and m.cols == 4
    for i in range(3):
        for j in range(4):
            assert abs(m[i, j] - arr[i, j]) < 1e-15


def test_matrix_from_numpy_different_sizes():
    """Test creating matrices from different sized NumPy arrays"""
    # 1x1
    arr1 = np.array([[5.0]], dtype=np.float64)
    m1 = Matrix(arr1)
    assert m1.rows == 1 and m1.cols == 1 and m1[0, 0] == 5.0

    # 1xN
    arr2 = np.array([[1, 2, 3, 4]], dtype=np.float64)
    m2 = Matrix(arr2)
    assert m2.rows == 1 and m2.cols == 4

    # Mx1
    arr3 = np.array([[1], [2], [3]], dtype=np.float64)
    m3 = Matrix(arr3)
    assert m3.rows == 3 and m3.cols == 1


def test_numpy_dtype_conversion():
    """Test that NumPy array is converted to float64"""
    # int array should be converted
    arr_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
    arr_float = arr_int.astype(np.float64)
    m = Matrix(arr_float)

    for i in range(2):
        for j in range(2):
            assert m[i, j] == float(arr_int[i, j])


def test_numpy_to_matrix_preserves_values():
    """Test that conversion preserves exact values"""
    values = [[1.5, 2.7, 3.9], [4.2, 5.8, 6.1]]
    arr = np.array(values, dtype=np.float64)
    m = Matrix(arr)

    for i in range(2):
        for j in range(3):
            assert m[i, j] == values[i][j]


def test_matrix_to_numpy_shape():
    """Test that Matrix → NumPy preserves shape"""
    m = Matrix(5, 7)
    arr = np.array(m)

    assert arr.shape == (5, 7)
    assert arr.dtype == np.float64


def test_numpy_operations_on_matrix():
    """Test NumPy operations work on matrix-backed arrays"""
    m = Matrix(np.array([[1, 2], [3, 4]], dtype=np.float64))
    arr = np.array(m)

    # NumPy operations should work
    result = arr + 10
    assert result[0, 0] == 11.0 and result[1, 1] == 14.0

    # Matrix multiplication
    arr2 = np.array([[5, 6], [7, 8]], dtype=np.float64)
    result2 = arr @ arr2
    assert result2.shape == (2, 2)


def test_matrix_equality_with_numpy():
    """Test matrix created from NumPy has same values"""
    arr = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64)
    m = Matrix(arr)

    # Element-wise comparison
    for i in range(2):
        for j in range(3):
            assert abs(m[i, j] - arr[i, j]) < 1e-15


def test_large_matrix_numpy_conversion():
    """Test conversion works for larger matrices"""
    arr = np.random.rand(100, 50)
    m = Matrix(arr)

    assert m.rows == 100 and m.cols == 50

    # Spot check a few elements
    assert abs(m[0, 0] - arr[0, 0]) < 1e-15
    assert abs(m[50, 25] - arr[50, 25]) < 1e-15
    assert abs(m[99, 49] - arr[99, 49]) < 1e-15


def test_matrix_operations_with_numpy_input():
    """Test matrix operations work with NumPy-created matrices"""
    arr1 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    arr2 = np.array([[5, 6], [7, 8]], dtype=np.float64)

    m1 = Matrix(arr1)
    m2 = Matrix(arr2)

    # Matrix operations
    m3 = m1 + m2
    expected = arr1 + arr2

    for i in range(2):
        for j in range(2):
            assert abs(m3[i, j] - expected[i, j]) < 1e-15


def test_numpy_contiguous_array():
    """Test that Matrix works with C-contiguous arrays"""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64, order="C")
    assert arr.flags["C_CONTIGUOUS"]

    m = Matrix(arr)
    assert m.rows == 2 and m.cols == 3


def test_matrix_shape_property():
    """Test Matrix.shape property for NumPy compatibility"""
    m = Matrix(3, 4)
    assert m.shape == (3, 4)

    m2 = Matrix.identity(5)
    assert m2.shape == (5, 5)

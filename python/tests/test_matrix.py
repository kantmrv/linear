"""
Test suite for Matrix class basic operations
"""

import numpy as np

from linear import Matrix


def test_matrix_creation():
    """Test Matrix construction from dimensions and NumPy arrays"""
    m = Matrix(3, 3)
    assert m.rows == 3 and m.cols == 3

    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    m = Matrix(arr)
    assert m[0, 0] == 1.0 and m[1, 1] == 4.0


def test_factory_methods():
    """Test identity, zeros, constant, random factories"""
    I = Matrix.identity(3)
    assert I[0, 0] == 1.0 and I[0, 1] == 0.0
    assert I[1, 1] == 1.0 and I[1, 2] == 0.0

    Z = Matrix.zeros(2, 3)
    assert Z.rows == 2 and Z.cols == 3
    assert Z[0, 0] == 0.0 and Z[1, 2] == 0.0

    C = Matrix.constant(2, 2, 5.0)
    assert C[0, 0] == 5.0 and C[1, 1] == 5.0

    R = Matrix.random(3, 3)
    assert R.rows == 3 and R.cols == 3
    # Random values should be in [-1, 1]
    for i in range(3):
        for j in range(3):
            assert -1.0 <= R[i, j] <= 1.0


def test_arithmetic():
    """Test +, -, *, / operators"""
    A = Matrix(np.array([[1, 2], [3, 4]], dtype=np.float64))
    B = Matrix(np.array([[5, 6], [7, 8]], dtype=np.float64))

    C = A + B
    assert C[0, 0] == 6.0 and C[0, 1] == 8.0
    assert C[1, 0] == 10.0 and C[1, 1] == 12.0

    D = A - B
    assert D[0, 0] == -4.0

    E = A * 2.0
    assert E[0, 0] == 2.0 and E[1, 1] == 8.0

    F = A / 2.0
    assert F[0, 0] == 0.5 and F[1, 1] == 2.0


def test_matrix_multiplication():
    """Test matrix multiplication"""
    A = Matrix(np.array([[1, 2], [3, 4]], dtype=np.float64))
    B = Matrix(np.array([[5, 6], [7, 8]], dtype=np.float64))

    C = A * B
    # [1*5+2*7, 1*6+2*8]   [19, 22]
    # [3*5+4*7, 3*6+4*8] = [43, 50]
    assert C[0, 0] == 19.0 and C[0, 1] == 22.0
    assert C[1, 0] == 43.0 and C[1, 1] == 50.0


def test_transpose():
    """Test transpose operation"""
    A = Matrix(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
    A.transpose()
    assert A.rows == 3 and A.cols == 2
    assert A[0, 0] == 1.0 and A[1, 0] == 2.0 and A[2, 0] == 3.0
    assert A[0, 1] == 4.0 and A[1, 1] == 5.0 and A[2, 1] == 6.0


def test_matrix_properties():
    """Test is_symmetric, is_diagonal, is_triangular"""
    I = Matrix.identity(3)
    assert I.is_symmetric()
    assert I.is_diagonal()
    assert I.is_upper_triangular()
    assert I.is_lower_triangular()

    # Symmetric but not diagonal
    S = Matrix(np.array([[1, 2], [2, 3]], dtype=np.float64))
    assert S.is_symmetric()
    assert not S.is_diagonal()

    # Upper triangular
    U = Matrix(np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float64))
    assert U.is_upper_triangular()
    assert not U.is_lower_triangular()


def test_norms():
    """Test frobenius, infinity, and euclidean norms"""
    A = Matrix(np.array([[3, 4]], dtype=np.float64))
    # Euclidean norm for row vector: sqrt(3^2 + 4^2) = 5
    assert abs(A.euclidean_norm() - 5.0) < 1e-10

    B = Matrix(np.array([[1, 2], [3, 4]], dtype=np.float64))
    # Infinity norm: max absolute row sum = max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
    assert B.infinity_norm() == 7.0

    # Frobenius norm: sqrt(sum of squares)
    # sqrt(1 + 4 + 9 + 16) = sqrt(30)
    assert abs(B.frobenius_norm() - np.sqrt(30.0)) < 1e-10


def test_element_access():
    """Test __getitem__ and __setitem__"""
    M = Matrix(2, 2)
    M[0, 0] = 1.0
    M[0, 1] = 2.0
    M[1, 0] = 3.0
    M[1, 1] = 4.0

    assert M[0, 0] == 1.0
    assert M[0, 1] == 2.0
    assert M[1, 0] == 3.0
    assert M[1, 1] == 4.0


def test_fill():
    """Test fill operation"""
    M = Matrix(3, 3)
    M.fill(7.0)

    for i in range(3):
        for j in range(3):
            assert M[i, j] == 7.0


def test_resize():
    """Test resize operation"""
    M = Matrix(2, 2)
    M.resize(3, 4)

    assert M.rows == 3 and M.cols == 4

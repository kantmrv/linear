"""
Test suite for matrix decompositions (LU, QR) and related operations
"""

import numpy as np

from linear import Matrix


def test_lu_decomposition():
    """Verify PA = LU for known matrix"""
    arr = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=np.float64)
    A = Matrix(arr)
    P, L, U = A.lu()

    assert L.is_lower_triangular()
    assert U.is_upper_triangular()

    # Verify PA = LU (within tolerance)
    PA = P * A
    LU = L * U
    for i in range(3):
        for j in range(3):
            assert abs(PA[i, j] - LU[i, j]) < 1e-10


def test_qr_decomposition():
    """Verify A = QR and Q is orthogonal"""
    A = Matrix(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64))
    Q, R = A.qr()

    assert R.is_upper_triangular()
    assert Q.is_orthogonal()

    # Verify A = QR
    QR = Q * R
    for i in range(3):
        for j in range(2):
            assert abs(A[i, j] - QR[i, j]) < 1e-10


def test_qr_square_matrix():
    """Test QR on square matrix"""
    A = Matrix(np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float64))
    Q, R = A.qr()

    assert Q.is_orthogonal()
    assert R.is_upper_triangular()

    # Verify A = QR
    QR = Q * R
    for i in range(3):
        for j in range(3):
            assert abs(A[i, j] - QR[i, j]) < 1e-9


def test_determinant():
    """Test determinant on known matrices"""
    # 2x2 matrix: det = ad - bc
    A = Matrix(np.array([[2, 3], [1, 4]], dtype=np.float64))
    det = A.lu_det()
    assert abs(det - 5.0) < 1e-10  # 2*4 - 3*1 = 5

    # Identity matrix
    I = Matrix.identity(3)
    assert abs(I.lu_det() - 1.0) < 1e-10

    # Diagonal matrix
    D = Matrix(np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.float64))
    assert abs(D.lu_det() - 24.0) < 1e-10  # 2*3*4 = 24


def test_matrix_inverse():
    """Verify A * A_inv = I"""
    A = Matrix(np.array([[4, 7], [2, 6]], dtype=np.float64))
    A_inv = A.lu_inv()

    I = A * A_inv
    for i in range(2):
        for j in range(2):
            expected = 1.0 if i == j else 0.0
            assert abs(I[i, j] - expected) < 1e-10


def test_inverse_3x3():
    """Test inverse on 3x3 matrix"""
    A = Matrix(np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=np.float64))
    A_inv = A.lu_inv()

    # Verify A * A_inv = I
    I = A * A_inv
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert abs(I[i, j] - expected) < 1e-9


def test_rank():
    """Test rank computation"""
    # Full rank
    A = Matrix.identity(3)
    assert A.qr_rank() == 3

    # Rank deficient - third row is 2*first row
    arr = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]], dtype=np.float64)
    B = Matrix(arr)
    rank = B.qr_rank()
    assert rank == 2  # Only 2 independent rows


def test_rank_column_matrix():
    """Test rank on tall matrix"""
    # 4x2 matrix with rank 2
    A = Matrix(np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float64))
    assert A.qr_rank() == 2


def test_condition_number():
    """Test condition number computation"""
    # Well-conditioned identity matrix
    I = Matrix.identity(3)
    cond = I.lu_cond()
    assert abs(cond - 1.0) < 1e-10

    # Ill-conditioned matrix
    A = Matrix(np.array([[1, 1], [1, 1.0001]], dtype=np.float64))
    cond = A.lu_cond()
    # Should have high condition number
    assert cond > 100


def test_forward_substitution():
    """Test forward substitution for lower triangular system"""
    # Lower triangular matrix
    L = Matrix(np.array([[2, 0, 0], [1, 3, 0], [-1, 2, 4]], dtype=np.float64))
    b = Matrix(np.array([[4], [7], [5]], dtype=np.float64))

    x = L.forward_substitution(b)

    # Verify Lx = b
    result = L * x
    for i in range(3):
        assert abs(result[i, 0] - b[i, 0]) < 1e-10


def test_backward_substitution():
    """Test backward substitution for upper triangular system"""
    # Upper triangular matrix
    U = Matrix(np.array([[2, 1, 3], [0, 3, 2], [0, 0, 4]], dtype=np.float64))
    b = Matrix(np.array([[12], [11], [8]], dtype=np.float64))

    x = U.backward_substitution(b)

    # Verify Ux = b
    result = U * x
    for i in range(3):
        assert abs(result[i, 0] - b[i, 0]) < 1e-10

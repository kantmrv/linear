"""
Test suite for Span class (linear subspace operations)
"""

import numpy as np

from linear import Matrix, Span


def test_span_creation():
    """Test basic span construction"""
    arr = np.array([[1, 0], [0, 1]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    assert span.dimension() == 2


def test_span_from_single_vector():
    """Test span from single vector"""
    arr = np.array([[1, 2, 3]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    assert span.dimension() == 1


def test_span_removes_dependent():
    """Test that linearly dependent vectors are removed"""
    # Third row = first + second
    arr = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    # Only 2 independent vectors
    assert span.dimension() == 2


def test_span_removes_multiple_dependent():
    """Test multiple dependent vectors are removed"""
    # Only first two rows are independent
    arr = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    assert span.dimension() == 2


def test_span_contains():
    """Test membership checking"""
    arr = np.array([[1, 0], [0, 1]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    # Vector in span
    v = Matrix(np.array([[1, 1]], dtype=np.float64))
    assert span.contains(v)

    # Another vector in span
    v2 = Matrix(np.array([[3, -2]], dtype=np.float64))
    assert span.contains(v2)


def test_span_contains_basis_vectors():
    """Test that basis vectors are contained in span"""
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    # Original vectors should be in span
    v1 = Matrix(np.array([[1, 2]], dtype=np.float64))
    v2 = Matrix(np.array([[3, 4]], dtype=np.float64))

    assert span.contains(v1)
    assert span.contains(v2)


def test_span_contains_matrix():
    """Test contains() with matrix argument"""
    arr = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    # Matrix with rows in span
    test_matrix = Matrix(np.array([[1, 1, 0], [2, -1, 0]], dtype=np.float64))
    assert span.contains(test_matrix)


def test_orthonormal_basis():
    """Test QR-based orthonormalization"""
    arr = np.array([[1, 1], [1, 0]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    Q = span.orthonormal_basis()
    assert Q.is_orthogonal()
    assert Q.rows == 2 and Q.cols == 2


def test_orthonormal_basis_3d():
    """Test orthonormal basis on 3D vectors"""
    arr = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    Q = span.orthonormal_basis()
    assert Q.is_orthogonal()
    assert Q.rows == 3


def test_span_random():
    """Test random vector generation in span"""
    arr = np.array([[1, 0], [0, 1]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    v = span.random()
    assert v.rows == 1 and v.cols == 2
    assert span.contains(v)


def test_span_random_multiple():
    """Test multiple random vectors are all in span"""
    arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    for _ in range(5):
        v = span.random()
        assert span.contains(v)


def test_empty_span():
    """Test empty span creation"""
    span = Span()
    assert span.dimension() == 0


def test_span_basis():
    """Test accessing basis matrix"""
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    basis = span.basis()
    assert basis.rows == 2 and basis.cols == 2


def test_span_insert():
    """Test inserting vectors into span"""
    arr = np.array([[1, 0, 0]], dtype=np.float64)
    m = Matrix(arr)
    span = Span(m)

    assert span.dimension() == 1

    # Insert linearly independent vector
    v = Matrix(np.array([[0, 1, 0]], dtype=np.float64))
    span.insert(v)

    assert span.dimension() == 2

    # Insert dependent vector (should not increase dimension)
    v_dep = Matrix(np.array([[2, 0, 0]], dtype=np.float64))
    span.insert(v_dep)

    assert span.dimension() == 2  # Dimension unchanged

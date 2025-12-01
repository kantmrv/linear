"""
Pytest configuration and shared fixtures
"""

import numpy as np
import pytest

from linear import Matrix


@pytest.fixture
def identity_3x3():
    """3x3 identity matrix fixture"""
    return Matrix.identity(3)


@pytest.fixture
def sample_2x2():
    """Sample 2x2 matrix fixture"""
    return Matrix(np.array([[1, 2], [3, 4]], dtype=np.float64))


@pytest.fixture
def sample_3x3():
    """Sample 3x3 matrix fixture"""
    return Matrix(np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=np.float64))


@pytest.fixture
def diag_dominant_3x3():
    """Diagonally dominant 3x3 matrix for iterative solvers"""
    return Matrix(np.array([[5, 1, 0], [1, 4, 1], [0, 1, 3]], dtype=np.float64))


@pytest.fixture
def symmetric_matrix():
    """Symmetric matrix fixture"""
    return Matrix(np.array([[4, 1, 2], [1, 5, 1], [2, 1, 6]], dtype=np.float64))


@pytest.fixture
def upper_triangular():
    """Upper triangular matrix fixture"""
    return Matrix(np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float64))


@pytest.fixture
def lower_triangular():
    """Lower triangular matrix fixture"""
    return Matrix(np.array([[2, 0, 0], [1, 3, 0], [-1, 2, 4]], dtype=np.float64))


@pytest.fixture
def column_vector():
    """Column vector (3x1 matrix) fixture"""
    return Matrix(np.array([[1], [2], [3]], dtype=np.float64))


@pytest.fixture
def row_vector():
    """Row vector (1x3 matrix) fixture"""
    return Matrix(np.array([[1, 2, 3]], dtype=np.float64))

# python/linear/__init__.py

from __future__ import annotations

__version__ = "0.1.0"

# Import the C++ extension module
try:
    from . import _linear_impl
except ImportError as e:
    raise ImportError(
        "Failed to import native extension '_linear_impl'. "
        "Make sure the C++ extension is built and installed."
    ) from e


class Matrix(_linear_impl.Matrix):
    @classmethod
    def identity(cls, n: int) -> "Matrix":
        """Create an n×n identity matrix."""
        return _linear_impl.Matrix.identity(n)

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        """Create a matrix filled with zeros."""
        return _linear_impl.Matrix.zeros(rows, cols)

    @classmethod
    def constant(cls, rows: int, cols: int, value: float) -> "Matrix":
        """Create a matrix filled with a constant value."""
        return _linear_impl.Matrix.constant(rows, cols, value)

    @classmethod
    def random(cls, rows: int, cols: int) -> "Matrix":
        """Create a matrix with random values in [-1, 1]."""
        return _linear_impl.Matrix.random(rows, cols)


# Aliases for solvers & Span
LUSolver = _linear_impl.LUSolver
QRSolver = _linear_impl.QRSolver
JacobianSolver = _linear_impl.JacobianSolver
GaussSeidelSolver = _linear_impl.GaussSeidelSolver
Span = _linear_impl.Span


__all__ = [
    "__version__",
    "Matrix",
    "LUSolver",
    "QRSolver",
    "JacobianSolver",
    "GaussSeidelSolver",
    "Span",
]

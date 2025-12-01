# Linear Algebra Python Package

Python bindings for a modern C++20 linear algebra library featuring compile-time type safety, comprehensive decompositions, and seamless NumPy integration.

## Overview

This package provides Python access to a high-performance C++ linear algebra library with:
- **Matrix operations** with LU and QR decompositions
- **Direct solvers** (LU, QR) and **iterative solvers** (Jacobi, Gauss-Seidel)
- **Linear subspaces** with automatic independence checking
- **NumPy integration** via the buffer protocol (zero-copy when possible)

## Installation

```bash
# Install from source in development mode
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from linear import Matrix, LUSolver, Span

# Create matrices
A = Matrix.identity(3)
B = Matrix(np.array([[4, 1], [1, 3]], dtype=np.float64))

# Matrix operations
C = A + A  # Addition
D = A * 2.0  # Scalar multiplication

# Solve linear system Ax = b
A = Matrix(np.array([[3, 2], [1, 2]], dtype=np.float64))
b = Matrix(np.array([[5], [3]], dtype=np.float64))

solver = LUSolver()
x = solver.solve(A, b)

# Matrix decompositions
P, L, U = A.lu()  # LU with partial pivoting
Q, R = A.qr()     # QR using Householder reflections

# Work with linear subspaces
basis = Matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64))
span = Span(basis)  # Automatically removes dependent vectors
print(f"Dimension: {span.dimension()}")  # 2
```

## API Reference

### Matrix Class

#### Constructors
- `Matrix()` - Create empty matrix
- `Matrix(rows, cols)` - Create zero-initialized matrix with given dimensions
- `Matrix(numpy_array)` - Create matrix from NumPy array (copies data)

#### Factory Methods (Classmethods)
- `Matrix.identity(n)` - Create n×n identity matrix
- `Matrix.zeros(rows, cols)` - Create matrix filled with zeros
- `Matrix.constant(rows, cols, value)` - Create matrix filled with constant value
- `Matrix.random(rows, cols)` - Create matrix with random values in [-1, 1]

#### Properties
- `rows` - Number of rows (read-only)
- `cols` - Number of columns (read-only)
- `size` - Total number of elements (read-only)
- `shape` - Tuple (rows, cols) for NumPy compatibility (read-only)

#### Element Access
- `matrix[i, j]` - Get element at position (i, j)
- `matrix[i, j] = value` - Set element at position (i, j)
- `matrix(i, j)` - Alternative element access (callable interface)

#### Matrix Decompositions
- `lu()` ’ `(P, L, U)` - LU decomposition with partial pivoting where PA = LU
- `qr()` ’ `(Q, R)` - QR decomposition using Householder reflections where A = QR
- `qr_rank()` - Compute matrix rank via QR decomposition
- `lu_det()` - Compute determinant via LU decomposition
- `lu_inv()` - Compute matrix inverse via LU decomposition
- `lu_cond()` - Compute condition number ||A|| × ||A{¹||

#### Matrix Properties & Predicates
- `is_symmetric()` - Check if matrix is symmetric (A = A@)
- `is_diagonal()` - Check if matrix is diagonal
- `is_upper_triangular()` - Check if matrix is upper triangular
- `is_lower_triangular()` - Check if matrix is lower triangular
- `is_orthogonal()` - Check if matrix is orthogonal (Q@Q = I)

#### Norms
- `frobenius_norm()` - Frobenius norm (|ab||²)
- `infinity_norm()` - Infinity norm (max absolute value)
- `euclidean_norm()` - Euclidean norm for vectors (L2 norm)

#### Matrix Operations
- `transpose()` - Transpose matrix in-place (returns self)
- `transposed()` - Return transposed copy of matrix
- `submatrix(row_offset, col_offset, rows, cols)` - Extract submatrix
- `set_submatrix(start_row, start_col, submatrix)` - Set block from another matrix
- `fill(value)` - Fill all elements with given value
- `resize(rows, cols)` - Resize matrix to new dimensions

#### Triangular System Solvers
- `forward_substitution(b)` - Solve Lx = b for lower triangular L
- `backward_substitution(b)` - Solve Ux = b for upper triangular U

#### Vector Operations (for row vectors)
- `dot(other)` - Dot product of two row vectors
- `normalize()` - Normalize vector in-place (returns self)
- `normalized()` - Return normalized copy of vector

#### Arithmetic Operators
- `A + B` - Matrix addition
- `A - B` - Matrix subtraction
- `A * B` - Matrix multiplication
- `A * scalar` - Scalar multiplication
- `scalar * A` - Scalar multiplication (commutative)
- `A / scalar` - Scalar division
- `-A` - Unary negation

#### Comparison Operators
- `A == B` - Element-wise equality
- `A != B` - Element-wise inequality

---

### Solver Classes

All solvers inherit from `Base_Solver` and provide the `solve(A, b)` method.

**Usage pattern**:
```python
solver = SolverClass()
x = solver.solve(A, b)
```

#### LUSolver
Direct method using LU decomposition with partial pivoting.

```python
solver = LUSolver()
x = solver.solve(A, b)
```

**Characteristics**:
- O(n³) complexity
- Robust for most systems
- Exact solution (within numerical precision)

#### QRSolver
Direct method using QR decomposition (more numerically stable).

```python
solver = QRSolver()
x = solver.solve(A, b)
```

**Characteristics**:
- O(n³) complexity
- Better numerical stability than LU
- Preferred for ill-conditioned systems

#### JacobianSolver
Iterative Jacobi method (requires diagonal dominance).

```python
solver = JacobianSolver(max_iterations=1000, tolerance=1e-10)
x = solver.solve(A, b)
```

**Parameters**:
- `max_iterations` - Maximum number of iterations (default: 1000)
- `tolerance` - Convergence tolerance (default: 1e-10)

**Properties** (read/write):
- `solver.max_iterations`
- `solver.tolerance`

**Convergence**:
Requires diagonally dominant or positive definite matrices: |abb| > |ab|| for j`i

#### GaussSeidelSolver
Iterative Gauss-Seidel method (faster convergence than Jacobi).

```python
solver = GaussSeidelSolver(max_iterations=500, tolerance=1e-10)
x = solver.solve(A, b)
```

**Characteristics**:
- Typically converges faster than Jacobi
- Uses updated values immediately (forward substitution-like)
- Same convergence requirements as Jacobi

---

### Span Class

Represents the linear span of a set of vectors with automatic linear independence checking.

#### Constructor
- `Span()` - Create empty span
- `Span(matrix)` - Create span from matrix rows (extracts linearly independent rows)

#### Methods
- `dimension()` - Get the dimension (number of basis vectors)
- `basis()` - Get the basis matrix (read-only)
- `contains(vector)` - Check if vector is in the span
- `contains(matrix)` - Check if all matrix rows are in the span
- `insert(vector)` - Insert vector if linearly independent
- `orthonormal_basis()` - Compute orthonormal basis via QR decomposition
- `random()` - Generate random vector in the span

**Example**:
```python
# Create span (automatically removes dependent vectors)
basis = Matrix(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]], dtype=np.float64))
span = Span(basis)

print(span.dimension())  # 2 (third row is dependent)

# Check membership
v = Matrix(np.array([[1, 1, 2]], dtype=np.float64))
print(span.contains(v))  # True

# Generate random vector in span
rand_v = span.random()
print(span.contains(rand_v))  # True
```

---

## NumPy Integration

The Matrix class seamlessly integrates with NumPy via the buffer protocol.

### Matrix  NumPy

```python
arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
m = Matrix(arr)  # Copies data
```

### Matrix ’ NumPy

```python
m = Matrix.identity(3)
arr = np.array(m)  # Zero-copy view when possible
```

**Important**: The NumPy array `arr` is a view into the Matrix's internal storage. Modifying `arr` may modify the original Matrix.

### NumPy Operations

```python
m = Matrix.random(3, 3)
arr = np.array(m)

# NumPy operations work seamlessly
result = arr @ arr.T  # Matrix multiplication via NumPy
mean = np.mean(arr)   # Statistical operations
```

---

## Testing

```bash
# Run all tests
uv run pytest python/tests/

# Run with verbose output
uv run pytest python/tests/ -v

# Run specific test file
uv run python/tests/test_matrix.py -v
```

---

## Performance Notes

The C++ backend provides significant performance optimizations:

1. **Cache-friendly matrix multiplication** - Uses i-k-j loop order for better cache locality
2. **Move semantics** - Efficient handling of temporaries without unnecessary copies
3. **Static dispatch** - Compile-time resolution for type-safe operations
4. **BLAS-like performance** - Hand-optimized algorithms for common operations

---

## Differences from C++ API

1. **Dynamic matrices only** - Python bindings expose only `Matrix<double, DynamicSize, DynamicSize>`
2. **Scalar type fixed** - Only float64 (double precision) is supported
3. **Factory methods** - Exposed as classmethods: `Matrix.identity(n)` instead of static methods
4. **Solver instantiation** - Create solver instances: `solver = LUSolver(); x = solver.solve(A, b)`
5. **NumPy integration** - Additional `__array__` protocol for seamless NumPy interop

---

## Requirements

- Python 3.9+
- NumPy 2.0+
- C++20 compatible compiler (for building from source)
- CMake 3.18+ (for building from source)

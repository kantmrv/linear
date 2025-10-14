# Linear Algebra Library

A modern C++20 header-only library for numerical linear algebra, featuring compile-time type safety through concepts and support for both static and dynamic matrix dimensions.

## Features

### Matrix Operations
- **Static and Dynamic Matrices**: Choose between compile-time sized (for performance) or runtime-sized (for flexibility)
- **Type-Safe Operations**: C++20 concepts ensure compile-time verification of matrix dimensions and types
- **Comprehensive Linear Algebra**:
  - LU decomposition with partial pivoting
  - QR decomposition using Householder reflections
  - Matrix rank computation
  - Matrix inverse and determinant
  - Condition number estimation
  - Forward/backward substitution for triangular systems

### Linear System Solvers
- **Direct Methods**:
  - LU Solver: Robust solver using LU decomposition
  - QR Solver: Numerically stable QR-based solver
- **Iterative Methods**:
  - Jacobi Solver: Simple iterative method with convergence control
  - Gauss-Seidel Solver: Faster convergence for diagonally dominant systems
  - A priori and a posteriori error estimates

### Vector Space Utilities
- **Span Class**: Represents linear subspaces
  - Automatic basis extraction from matrices
  - Linear independence checking
  - QR-based orthonormalization (numerically stable)
  - Random vector generation in the span

## Requirements

- C++20 compatible compiler (Clang 14+, GCC 11+, MSVC 2022+)
- Standard library with concepts support

## Installation

This is a header-only library. Simply include the headers in your project:

```cpp
#include "Matrix.hpp"
#include "Solver.hpp"
#include "Span.hpp"
```

## Quick Start

### Creating Matrices

```cpp
using namespace linear;

// Static 3x3 matrix (compile-time size)
Matrix<double, 3, 3> A = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 10}
};

// Dynamic matrix (runtime size)
Matrix<double> B(3, 3);

// Factory methods
auto I = Matrix<double, 3, 3>::Identity();
auto zeros = Matrix<double>::Zeros(4, 4);
auto random = Matrix<double>::Random(5, 5, true);  // diagonal dominance
```

### Solving Linear Systems

```cpp
// Solve Ax = b using LU decomposition
Matrix<double> A(3, 3);
Matrix<double> b(3, 1);

LU_Solver<double> solver;
auto x = solver.solve(A, b);

// Or use the static convenience method
auto x = LU_Solver<double>::solve_system(A, b);

// Iterative solvers with convergence control
Jacobian_Solver<double> jacobi(1000, 1e-10);  // max_iter, tolerance
auto x = jacobi.solve(A, b);
```

### Matrix Decompositions

```cpp
Matrix<double, 3, 3> A = /* ... */;

// LU decomposition
auto [P, L, U] = A.lu();  // PA = LU

// QR decomposition
auto [Q, R] = A.qr();  // A = QR

// Matrix properties
double det = A.lu_det();
double cond = A.lu_cond();
size_t r = A.rank();
auto A_inv = A.lu_inv();
```

### Working with Spans

```cpp
// Create span from matrix rows
Matrix<double> basis = {
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 2}  // linearly dependent - will be excluded
};
Span<double> S(basis);

std::cout << "Dimension: " << S.dimension() << std::endl;  // 2

// Check containment
Vector<double> v = {1, 1, 2};
bool in_span = S.contains(v);  // true

// Generate random vector in span
auto random_v = S.random();

// Orthonormalize basis
auto Q = S.orthonormal_basis();
```

## Architecture

### Template Design

The library uses extensive template metaprogramming for compile-time safety:

```cpp
// Dimension parameters: positive for static, DynamicSize for dynamic
template<Numeric Scalar, ptrdiff_t Rows = DynamicSize, ptrdiff_t Cols = DynamicSize>
class Matrix { /* ... */ };

// Concepts enforce compatibility
template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
    requires(CompatibleForMultiplication<Cols, InRows>)
friend constexpr auto operator*(const Matrix& lhs, const Matrix<InScalar, InRows, InCols>& rhs);
```

### Storage Strategy

- **Static matrices**: `std::array<Scalar, Rows * Cols>` - stack allocated, cache-friendly
- **Dynamic matrices**: `std::vector<Scalar>` - heap allocated, resizable

### Performance Optimizations

- Move semantics for temporary objects
- Cache-friendly matrix multiplication (i-k-j loop order)
- Expression templates for rvalue optimization
- `constexpr` methods for compile-time evaluation where possible

## Build Instructions

```bash
# Compile main program
clang++ -std=c++20 main.cpp -o main

# With optimizations
clang++ -std=c++20 -O3 main.cpp -o main

# With warnings
clang++ -std=c++20 -Wall -Wextra main.cpp -o main
```

## API Reference

### Matrix Class

#### Constructors
- `Matrix()` - Default constructor
- `Matrix(size_t rows, size_t cols)` - Dynamic size constructor
- `Matrix(initializer_list...)` - Initialize from nested lists

#### Static Factory Methods
- `Identity()` / `Identity(n)` - Identity matrix
- `Zeros(r, c)` - Zero matrix
- `Ones(r, c)` - Matrix of ones
- `Random(r, c, diag_dom, symmetric, positive)` - Random matrix generation

#### Core Operations
- `operator+`, `operator-`, `operator*` - Arithmetic operations
- `operator()` - Element access
- `transposed()` / `transpose()` - Matrix transpose
- `submatrix(row, col, rows, cols)` - Extract submatrix
- `dot(other)` - Dot product (for vectors)

#### Linear Algebra
- `lu()` → `(P, L, U)` - LU decomposition
- `qr()` → `(Q, R)` - QR decomposition
- `rank()` - Compute rank
- `lu_det()` - Determinant
- `lu_inv()` - Matrix inverse
- `lu_cond()` - Condition number
- `forward_substitution(b)` - Solve Lx = b (L lower triangular)
- `backward_substitution(b)` - Solve Ux = b (U upper triangular)

### Solver Classes

#### Base_Solver
```cpp
virtual Matrix solve(const Matrix& A, const Matrix& b) const = 0;
```

#### Iter_Solver
```cpp
Iter_Solver(size_t max_iter = 1000, Scalar tolerance = 1e-10);
Scalar prior_estimate(const Matrix& A, size_t iterations) const;
Scalar posterior_estimate(const Matrix& A, const Matrix& x_curr, const Matrix& x_prev) const;
```

### Span Class

#### Methods
- `dimension()` - Number of basis vectors
- `basis()` - Get basis matrix
- `contains(vector)` - Check if vector is in span
- `insert(vector)` - Add linearly independent vector
- `orthonormal_basis()` - Compute basis
- `random()` - Generate random vector in span

## Type Aliases

```cpp
// Square matrices
using Matrix2d = MatrixX<double, 2>;
using Matrix3d = MatrixX<double, 3>;
using Matrix4d = MatrixX<double, 4>;

// Vectors (row vectors)
using Vector2d = VectorX<double, 2>;
using Vector3d = VectorX<double, 3>;
using Vector4d = VectorX<double, 4>;

// Dynamic types
using Vector<double> = Matrix<double, 1, DynamicSize>;
```

## Best Practices

1. **Use static matrices when size is known at compile time** for better performance
2. **Ensure diagonal dominance** for iterative solvers to guarantee convergence
3. **Check condition numbers** before solving systems - high values indicate ill-conditioning
4. **Use QR for least squares** problems - more numerically stable than normal equations
5. **Prefer move semantics** - use `std::move()` or rvalue expressions for large matrices

## Limitations

- Iterative solvers require diagonally dominant or positive definite matrices for guaranteed convergence
- QR decomposition produces economy-size factorization (m × n matrix → m × min(m,n) Q and min(m,n) × n R)
- Rank computation uses numerical tolerance - very small pivots are treated as zero

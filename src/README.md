# C++ Linear Algebra Headers

Modern C++20 header-only library for numerical linear algebra with compile-time type safety through concepts and support for both static (compile-time sized) and dynamic (runtime-sized) matrices.

## Overview

This library provides:
- **Compile-time dimension checking** via C++20 concepts
- **Dual storage strategy** - `std::array` for static matrices, `std::vector` for dynamic
- **Extensive linear algebra operations** - decompositions, solvers, norms
- **Template metaprogramming** - dimension parameters enable static/dynamic dispatch
- **Zero external dependencies** - header-only, uses only C++ standard library

---

## Files

### Matrix.hpp (~1400 lines)

Core matrix class with dual storage and comprehensive operations.

**Key features**:
- Template parameters: `Matrix<Scalar, Rows, Cols>`
- `DynamicSize` constant (`-1`) indicates runtime-determined size
- 91 public methods covering construction, decompositions, properties, operations
- LU and QR decompositions with partial pivoting/Householder reflections
- Rank, determinant, inverse, condition number computation
- Forward/backward substitution for triangular systems

**Storage**:
```cpp
using Storage = std::conditional_t<
    Rows != DynamicSize && Cols != DynamicSize,
    std::array<Scalar, Rows * Cols>,  // Static
    std::vector<Scalar>                // Dynamic
>;
```

**Example**:
```cpp
// Static 3×3 matrix (stack-allocated)
Matrix<double, 3, 3> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};

// Dynamic matrix (heap-allocated)
Matrix<double> B(3, 3);

// LU decomposition
auto [P, L, U] = A.lu();  // PA = LU
```

---

### Solver.hpp (~330 lines)

Solver hierarchy for linear systems Ax = b.

**Class hierarchy**:
```
Base_Solver<Scalar>  (abstract base)
  LU_Solver        - Direct LU decomposition
  QR_Solver        - Direct QR decomposition (more stable)
  Iter_Solver<Scalar>  (iterative base)
    Jacobian_Solver      - x^(k+1) = D^-1(b - (L+U)x^(k))
    Gauss_Seidel_Solver  - Uses updated values immediately
```

**Iterative solver features**:
- Configurable max_iterations and tolerance
- A priori error estimation: `||e^(k)|| d ||M||^k ||e^(0)||`
- A posteriori error estimation: `||e^(k)|| d ||M||/(1-||M||) ||x^(k) - x^(k-1)||`
- Requires diagonally dominant or positive definite matrices for convergence

**Example**:
```cpp
Matrix<double> A = /* ... */;
Matrix<double> b = /* ... */;

// Direct solver
LU_Solver<double> lu_solver;
auto x = lu_solver.solve(A, b);

// Iterative solver
Jacobian_Solver<double> jacobi(1000, 1e-10);  // max_iter, tolerance
auto x2 = jacobi.solve(A, b, x0);  // with initial guess
```

---

### Span.hpp (~150 lines)

Linear subspace representation with automatic independence checking.

**Key features**:
- Constructor extracts linearly independent rows from input matrix
- Uses rank-based independence checking via QR decomposition
- QR-based orthonormalization (numerically stable)
- Membership testing via least squares solution
- Random vector generation as linear combination of basis vectors

**Example**:
```cpp
Matrix<double> vectors = {{1, 0, 1}, {0, 1, 1}, {1, 1, 2}};  // 3rd dependent
Span<double> S(vectors);

S.dimension();  // 2 (only 2 independent vectors)

Vector<double> v = {1, 1, 2};
S.contains(v);  // true

auto Q = S.orthonormal_basis();  // QR-based orthonormalization
```

---

## Design Patterns

### Template Metaprogramming with Dimension Parameters

```cpp
template<Numeric Scalar, ptrdiff_t Rows = DynamicSize, ptrdiff_t Cols = DynamicSize>
class Matrix;

// Static 3×3 matrix
Matrix<double, 3, 3> static_matrix;

// Dynamic matrix
Matrix<double, DynamicSize, DynamicSize> dynamic_matrix(5, 5);
Matrix<double> dynamic_alias(5, 5);  // Equivalent
```

**Design rationale**:
- Positive integers ’ compile-time size (stack allocation)
- `DynamicSize` (-1) ’ runtime size (heap allocation)
- Type system enforces dimension compatibility at compile time

---

### C++20 Concepts for Type Safety

The library defines comprehensive concepts for compile-time verification:

**Numeric type concepts**:
```cpp
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;
```

**Matrix shape concepts**:
```cpp
template<ptrdiff_t Rows, ptrdiff_t Cols>
concept SquareMatrix = (Rows == Cols) && (Rows > 0);

template<ptrdiff_t Rows, ptrdiff_t Cols>
concept RowVector = (Rows == 1);

template<ptrdiff_t Rows, ptrdiff_t Cols>
concept ColumnVector = (Cols == 1);
```

**Dimension compatibility concepts**:
```cpp
template<ptrdiff_t Rows1, ptrdiff_t Rows2>
concept CompatibleForAddition =
    (Rows1 == DynamicSize) || (Rows2 == DynamicSize) || (Rows1 == Rows2);

template<ptrdiff_t Cols1, ptrdiff_t Rows2>
concept CompatibleForMultiplication =
    (Cols1 == DynamicSize) || (Rows2 == DynamicSize) || (Cols1 == Rows2);
```

**Storage type concepts**:
```cpp
template<ptrdiff_t Rows, ptrdiff_t Cols>
concept StaticMatrix = (Rows > 0) && (Cols > 0);

template<ptrdiff_t Rows, ptrdiff_t Cols>
concept DynamicMatrix = (Rows == DynamicSize) || (Cols == DynamicSize);
```

---

### Common Type Promotion

Operations between different scalar types use `std::common_type_t`:

```cpp
Matrix<int, 3, 3> A_int;
Matrix<double, 3, 3> B_double;

auto C = A_int + B_double;  // Result type: Matrix<double, 3, 3>
```

---

### Performance Optimizations

1. **Cache-friendly matrix multiplication**:
```cpp
// i-k-j loop order for better cache locality
for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < inner; ++k) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) += lhs(i, k) * rhs(k, j);
        }
    }
}
```

2. **Move semantics for temporaries**:
```cpp
Matrix<double, 3, 3> A, B, C;
auto result = A + B + C;  // Uses rvalue references, no extra copies
```

3. **constexpr for compile-time evaluation**:
```cpp
constexpr auto I = Matrix<double, 3, 3>::Identity();  // Compile-time
```

---

## Numerical Considerations

### 1. Rank Tolerance

Uses machine epsilon for rank determination:
```cpp
if (std::abs(R(i, i)) > std::numeric_limits<Scalar>::epsilon()) {
    ++rank_count;
}
```

---

### 2. Determinant Sign Calculation

Current implementation counts entries below diagonal in permutation matrix:
```cpp
size_t swaps = 0;
for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
        if (std::abs(P(i, j)) > epsilon) { ++swaps; }
    }
}
Scalar sign = (swaps % 2 == 0) ? Scalar(1) : Scalar(-1);
```

---

### 3. QR Decomposition

Produces economy-size factorization:
- Input: m×n matrix A
- Output: Q is m×min(m,n), R is min(m,n)×n

**Implication**: For m > n (tall matrices), Q is not square. For full QR, additional columns would be needed.

---

### 4. Iterative Solver Convergence

Convergence condition: spectral radius Á(M) < 1

**Sufficient condition**: Diagonal dominance
```
|a_ii| > £|a_ij| for j ` i (all rows)
```

**Alternative**: Symmetric positive definite matrices also guarantee convergence.

---

## Compilation

Requires C++20 with concepts support:

```bash
# Clang 14+
clang++ -std=c++20 main.cpp -o main

# GCC 11+
g++ -std=c++20 main.cpp -o main

# With optimizations
clang++ -std=c++20 -O3 -march=native main.cpp -o main

# With warnings
clang++ -std=c++20 -Wall -Wextra -Wpedantic main.cpp -o main
```

---

## Common Type Aliases

```cpp
namespace linear {
    // Square matrices (static)
    template<Numeric Scalar, ptrdiff_t N>
    using MatrixX = Matrix<Scalar, N, N>;

    using Matrix2d = MatrixX<double, 2>;  // 2×2
    using Matrix3d = MatrixX<double, 3>;  // 3×3
    using Matrix4d = MatrixX<double, 4>;  // 4×4

    // Vectors (row vectors)
    template<Numeric Scalar, ptrdiff_t Size>
    using VectorX = Matrix<Scalar, 1, Size>;

    using Vector2d = VectorX<double, 2>;
    using Vector3d = VectorX<double, 3>;
    using Vector4d = VectorX<double, 4>;

    // Dynamic types
    template<Numeric Scalar>
    using Vector = Matrix<Scalar, 1, DynamicSize>;
}
```

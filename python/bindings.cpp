//
//  python/bindings.cpp
//  linear
//
//  Created by Alexandr on 01.10.2025.
//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <sstream>

#include "Matrix.hpp"
#include "Solver.hpp"
#include "Span.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace linear;

// Forward declarations for binding functions
template <typename Scalar>
void bind_matrix(nb::module_& m, const char* name);

template <typename Scalar>
void bind_solvers(nb::module_& m, const char* suffix);

template <typename Scalar>
void bind_span(nb::module_& m, const char* name);

// Matrix bindings implementation
template <typename Scalar>
void bind_matrix(nb::module_& m, const char* name) {
    using Mat = Matrix<Scalar, DynamicSize, DynamicSize>;

    nb::class_<Mat>(m, name)
        // Constructors
        .def(nb::init<>(), "Default constructor (creates empty matrix)")

        .def(nb::init<size_t, size_t>(), "rows"_a, "cols"_a, "Create matrix with given dimensions (zero-initialized)")

        .def(nb::init<const Mat&>(), "other"_a, "Copy constructor")

        // NumPy array constructor
        .def(
            "__init__",
            [](Mat* m, nb::ndarray<Scalar, nb::ndim<2>, nb::c_contig> arr) {
                new (m) Mat(arr.shape(0), arr.shape(1));
                std::memcpy(m->data(), arr.data(), arr.size() * sizeof(Scalar));
            },
            "array"_a,
            "Create from NumPy array (copies data)")

        // Properties (read-only)
        .def_prop_ro("rows", &Mat::rows, "Number of rows")
        .def_prop_ro("cols", &Mat::cols, "Number of columns")
        .def_prop_ro("size", &Mat::size, "Total number of elements")
        .def_prop_ro(
            "shape",
            [](const Mat& m) { return std::make_pair(m.rows(), m.cols()); },
            "Shape as (rows, cols) tuple")

        // Element access
        .def(
            "__call__",
            [](Mat& m, size_t i, size_t j) -> Scalar& {
                if (i >= m.rows() || j >= m.cols()) {
                    throw std::out_of_range("Matrix index out of range");
                }
                return m(i, j);
            },
            "i"_a,
            "j"_a,
            nb::rv_policy::reference_internal,
            "Access element at (i, j)")

        .def(
            "__call__",
            [](const Mat& m, size_t i, size_t j) -> Scalar {
                if (i >= m.rows() || j >= m.cols()) {
                    throw std::out_of_range("Matrix index out of range");
                }
                return m(i, j);
            },
            "i"_a,
            "j"_a,
            "Access element at (i, j) (const)")

        .def(
            "__getitem__",
            [](const Mat& m, std::pair<size_t, size_t> idx) {
                if (idx.first >= m.rows() || idx.second >= m.cols()) {
                    throw std::out_of_range("Matrix index out of range");
                }
                return m(idx.first, idx.second);
            },
            "index"_a,
            "Get element at [i, j]")

        .def(
            "__setitem__",
            [](Mat& m, std::pair<size_t, size_t> idx, Scalar val) {
                if (idx.first >= m.rows() || idx.second >= m.cols()) {
                    throw std::out_of_range("Matrix index out of range");
                }
                m(idx.first, idx.second) = val;
            },
            "index"_a,
            "value"_a,
            "Set element at [i, j]")

        .def(
            "data",
            [](Mat& m) { return m.data(); },
            nb::rv_policy::reference_internal,
            "Get pointer to underlying data (row-major order)")

        // NumPy array interface
        .def_prop_ro("__array_interface__",
                     [](Mat& m) -> nb::dict {
                         nb::dict result;
                         result["version"] = 3;
                         result["typestr"] = (sizeof(Scalar) == 8) ? "<f8" : "<f4";
                         result["data"] = nb::make_tuple(reinterpret_cast<uintptr_t>(m.data()),
                                                         false); // (data pointer, read-only flag)
                         result["shape"] = nb::make_tuple(m.rows(), m.cols());
                         result["strides"] = nb::make_tuple(m.cols() * sizeof(Scalar), sizeof(Scalar));
                         return result;
                     })

        // Factory methods (static)
        .def_static("identity", &Mat::Identity, "n"_a, "Create n×n identity matrix")

        .def_static("zeros", &Mat::Zeros, "rows"_a, "cols"_a, "Create matrix filled with zeros")

        .def_static("constant",
                    &Mat::Constant,
                    "rows"_a,
                    "cols"_a,
                    "value"_a,
                    "Create matrix filled with constant value")

        .def_static("random", &Mat::Random, "rows"_a, "cols"_a, "Create matrix with random values in [-1, 1]")

        // Operations
        .def("fill", &Mat::fill, "value"_a, "Fill all elements with given value")

        .def("resize", &Mat::resize, "rows"_a, "cols"_a, "Resize matrix to new dimensions")

        .def("submatrix", &Mat::submatrix, "row_offset"_a, "col_offset"_a, "rows"_a, "cols"_a, "Extract submatrix")

        .def("set_submatrix",
             &Mat::template set_submatrix<double, DynamicSize, DynamicSize>,
             "start_row"_a,
             "start_col"_a,
             "submatrix"_a,
             "Set a block of the matrix from another matrix")

        .def("transpose", &Mat::transpose, "Transpose matrix in-place (returns self)")

        .def("transposed", &Mat::transposed, "Return transposed copy of matrix")

        // Matrix manipulation
        .def(
            "insert",
            [](Mat& m, const Mat& other) { m.insert(other); },
            "other"_a,
            "Append rows from another matrix")

        // Linear algebra operations
        .def("lu",
             &Mat::lu,
             "LU decomposition with partial pivoting. Returns (P, L, U) where PA "
             "= LU")

        .def("qr", &Mat::qr, "QR decomposition using Householder reflections. Returns (Q, R)")

        .def("qr_rank", &Mat::qr_rank, "Compute matrix rank via QR decomposition")

        .def("lu_det", &Mat::lu_det, "Compute determinant via LU decomposition")

        .def("lu_inv", &Mat::lu_inv, "Compute matrix inverse via LU decomposition")

        .def("lu_cond", &Mat::lu_cond, "Compute condition number ||A|| * ||A^-1||")

        // Matrix property checks
        .def("is_symmetric", &Mat::isSymmetric, "Check if matrix is symmetric (uses epsilon tolerance)")

        .def("is_diagonal", &Mat::isDiagonal, "Check if matrix is diagonal (uses epsilon tolerance)")

        .def("is_upper_triangular",
             &Mat::isUpperTriangular,
             "Check if matrix is upper triangular (uses epsilon tolerance)")

        .def("is_lower_triangular",
             &Mat::isLowerTriangular,
             "Check if matrix is lower triangular (uses epsilon tolerance)")

        .def("is_orthogonal", &Mat::isOrthogonal, "Check if matrix is orthogonal Q^T * Q = I (uses epsilon tolerance)")

        // Norm methods
        .def("frobenius_norm", &Mat::frobenius_norm, "Compute Frobenius norm (square root of sum of squared elements)")

        .def("infinity_norm", &Mat::infinity_norm, "Compute infinity norm (max absolute value of any element)")

        .def("euclidean_norm", &Mat::euclidean_norm, "Compute Euclidean norm (L2 norm) for vectors")
        // Triangular system solvers
        .def("forward_substitution",
             &Mat::forward_substitution,
             "b"_a,
             "Solve Lx = b where L is lower triangular with unit diagonal")

        .def("backward_substitution", &Mat::backward_substitution, "b"_a, "Solve Ux = b where U is upper triangular")

        // Vector operations (for row vectors)
        .def("dot", &Mat::template dot<double, DynamicSize, DynamicSize>, "other"_a, "Dot product of two row vectors")

        .def("normalize", &Mat::normalize, "Normalize vector in-place (returns self)")

        .def("normalized", &Mat::normalized, "Return normalized copy of vector")

        // Arithmetic operators
        .def(nb::self + nb::self, "Matrix addition")
        .def(nb::self - nb::self, "Matrix subtraction")
        .def(nb::self * nb::self, "Matrix multiplication")
        .def(nb::self * double(), "Scalar multiplication (matrix * scalar)")
        .def(double() * nb::self, "Scalar multiplication (scalar * matrix)")
        .def(nb::self / double(), "Scalar division")
        .def(-nb::self, "Unary negation")

        // Comparison operators
        .def(nb::self == nb::self, "Equality comparison")
        .def(nb::self != nb::self, "Inequality comparison")

        // String representation
        .def("__repr__",
             [](const Mat& m) {
                 std::ostringstream oss;
                 oss << "Matrix(" << m.rows() << "×" << m.cols() << ")";
                 return oss.str();
             })

        .def("__str__", [](const Mat& m) {
            std::ostringstream oss;
            oss << m;
            return oss.str();
        });
}

// Solver bindings
template <typename Scalar>
void bind_solvers(nb::module_& m, const char* suffix) {
    using Mat = Matrix<Scalar, DynamicSize, DynamicSize>;
    std::string s(suffix);

    // Base solver (abstract)
    nb::class_<Base_Solver<Scalar>>(m, ("BaseSolver" + s).c_str())
        .def("solve", &Base_Solver<Scalar>::solve, "A"_a, "b"_a, "Solve linear system Ax = b");
    // Iterative solver base (abstract - no constructor)
    nb::class_<Iter_Solver<Scalar>, Base_Solver<Scalar>>(m, ("IterSolver" + s).c_str())
        .def("solve",
             nb::overload_cast<const Mat&, const Mat&>(&Iter_Solver<Scalar>::solve, nb::const_),
             "A"_a,
             "b"_a,
             "Solve Ax = b with zero initial guess")
        .def("solve",
             nb::overload_cast<const Mat&, const Mat&, const Mat&>(&Iter_Solver<Scalar>::solve, nb::const_),
             "A"_a,
             "b"_a,
             "x0"_a,
             "Solve Ax = b with initial guess x0")

        .def_prop_rw("max_iterations",
                     &Iter_Solver<Scalar>::max_iterations,
                     &Iter_Solver<Scalar>::set_max_iterations,
                     "Maximum number of iterations")

        .def_prop_rw("tolerance",
                     &Iter_Solver<Scalar>::tolerance,
                     &Iter_Solver<Scalar>::set_tolerance,
                     "Convergence tolerance")

        .def("prior_estimate", &Iter_Solver<Scalar>::prior_estimate, "A"_a, "iterations"_a, "A priori error estimate")

        .def("posterior_estimate",
             &Iter_Solver<Scalar>::posterior_estimate,
             "A"_a,
             "x_curr"_a,
             "x_prev"_a,
             "A posteriori error estimate");
    // LU Solver
    nb::class_<LU_Solver<Scalar>, Base_Solver<Scalar>>(m, ("LUSolver" + s).c_str())
        .def(nb::init<>(), "Create LU solver")
        .def("solve", &LU_Solver<Scalar>::solve, "A"_a, "b"_a, "Solve Ax = b using LU decomposition");

    // QR Solver
    nb::class_<QR_Solver<Scalar>, Base_Solver<Scalar>>(m, ("QRSolver" + s).c_str())
        .def(nb::init<>(), "Create QR solver")
        .def("solve", &QR_Solver<Scalar>::solve, "A"_a, "b"_a, "Solve Ax = b using QR decomposition");

    // Jacobi Solver
    nb::class_<Jacobian_Solver<Scalar>, Iter_Solver<Scalar>>(m, ("JacobianSolver" + s).c_str())
        .def(nb::init<size_t, Scalar>(),
             "max_iterations"_a = 1000,
             "tolerance"_a = Scalar(1e-10),
             "Create Jacobi iterative solver");

    // Gauss-Seidel Solver
    nb::class_<Gauss_Seidel_Solver<Scalar>, Iter_Solver<Scalar>>(m, ("GaussSeidelSolver" + s).c_str())
        .def(nb::init<size_t, Scalar>(),
             "max_iterations"_a = 1000,
             "tolerance"_a = Scalar(1e-10),
             "Create Gauss-Seidel iterative solver");
}

// Span bindings
template <typename Scalar>
void bind_span(nb::module_& m, const char* name) {
    using SpanT = Span<Scalar>;
    using Vec = Vector<Scalar>;
    using Mat = Matrix<Scalar, DynamicSize, DynamicSize>;

    nb::class_<SpanT>(m, name)
        .def(nb::init<>(), "Create empty span")

        .def(nb::init<const Mat&>(), "matrix"_a, "Create span from matrix rows (extracts linearly independent rows)")

        .def(
            "insert",
            [](SpanT& span, const Mat& v) { span.insert(v); },
            "vector"_a,
            "Insert row vector (1×n matrix) if linearly independent")

        .def(
            "contains",
            [](const SpanT& span, const Vec& v) { return span.contains(v); },
            "vector"_a,
            "Check if vector is in the span")

        .def(
            "contains",
            [](const SpanT& span, const Mat& m) { return span.contains(m); },
            "matrix"_a,
            "Check if all matrix rows are in the span")

        .def("dimension", &SpanT::dimension, "Get the dimension (number of basis vectors)")

        .def("basis", &SpanT::basis, nb::rv_policy::reference_internal, "Get the basis matrix")

        .def("orthonormal_basis", &SpanT::orthonormal_basis, "Compute orthonormal basis via QR decomposition")

        .def(
            "random",
            [](const SpanT& span) -> Mat {
                Vec v = span.random();
                // Convert Vector (1 x n) to Matrix (DynamicSize x DynamicSize)
                Mat result(v.rows(), v.cols());
                for (size_t j = 0; j < v.cols(); ++j) {
                    result(0, j) = v(0, j);
                }
                return result;
            },
            "Generate random vector in the span");
}

// Module definition
NB_MODULE(_linear_impl, m) {
    m.doc() = "C++ linear algebra library Python bindings";

    // Bind double precision (primary)
    bind_matrix<double>(m, "Matrix");
    bind_solvers<double>(m, "");
    bind_span<double>(m, "Span");
}

//
//  Matrix.hpp
//  linear
//
//  Created by Alexandr on 01.10.2025.
//

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <ostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


namespace linear {
    constexpr ptrdiff_t DynamicSize = -1;

    // Type safety concepts
    template<typename T>
    concept Numeric = std::is_arithmetic_v<T>;

    template<typename T>
    concept FloatingPoint = std::floating_point<T>;

    template<typename T>
    concept Integral = std::integral<T>;

    // Matrix dimension concepts
    template<ptrdiff_t R1, ptrdiff_t C1, ptrdiff_t R2, ptrdiff_t C2>
    concept CompatibleForAddition =
        (R1 == R2 or R1 == DynamicSize or R2 == DynamicSize) and
        (C1 == C2 or C1 == DynamicSize or C2 == DynamicSize);

    template<ptrdiff_t C1, ptrdiff_t R2>
    concept CompatibleForMultiplication =
        (C1 == R2) or (C1 == DynamicSize) or (R2 == DynamicSize);

    template<ptrdiff_t R, ptrdiff_t C>
    concept SquareMatrix = (R == C) and (R != DynamicSize);

    template<ptrdiff_t R, ptrdiff_t C>
    concept RowVector = (R == 1) or (R == DynamicSize);

    template<ptrdiff_t R, ptrdiff_t C>
    concept ColumnVector = (C == 1) or (C == DynamicSize);

    template<ptrdiff_t R, ptrdiff_t C>
    concept StaticMatrix = (R != DynamicSize) and (C != DynamicSize);

    template<ptrdiff_t R, ptrdiff_t C>
    concept DynamicMatrix = (R == DynamicSize) or (C == DynamicSize);

    template<Numeric Scalar_, ptrdiff_t Rows_ = DynamicSize, ptrdiff_t Cols_ = DynamicSize>
    class Matrix {
    public:
        using Scalar = Scalar_;
        inline static constexpr ptrdiff_t Rows = Rows_;
        inline static constexpr ptrdiff_t Cols = Cols_;

        inline static constexpr bool isRowsDynamic = (Rows == DynamicSize);
        inline static constexpr bool isColsDynamic = (Cols == DynamicSize);
        inline static constexpr bool isDynamic = (isRowsDynamic or isColsDynamic);

    private:
        template<Numeric InScalar>
        using CommonScalar = std::common_type_t<Scalar, InScalar>;
        
        using StaticContainer = std::array<Scalar, (isDynamic ? 1 : Rows * Cols)>;
        using DynamicContainer = std::vector<Scalar>;
        using ContainerType = std::conditional_t<isDynamic, DynamicContainer, StaticContainer>;

        size_t        m_rows;
        size_t        m_cols;
        ContainerType m_container;
        
        // Helper methods for linear algebra operations

        // Sign function for numerical algorithms
        static constexpr Scalar sign(Scalar x) noexcept {
            if (x > Scalar(0)) return Scalar(1);
            if (x < Scalar(0)) return Scalar(-1);
            return Scalar(0);
        }

        // Infinity norm (maximum absolute row sum)
        constexpr Scalar infinity_norm() const noexcept {
            Scalar max_sum = Scalar(0);
            for (size_t i = 0; i < m_rows; ++i) {
                Scalar row_sum = Scalar(0);
                for (size_t j = 0; j < m_cols; ++j) {
                    row_sum += std::abs((*this)(i, j));
                }
                max_sum = std::max(max_sum, row_sum);
            }
            return max_sum;
        }

        // Swap two rows in the matrix
        constexpr void swap_rows(size_t row1, size_t row2) noexcept {
            if (row1 == row2) return;
            for (size_t j = 0; j < m_cols; ++j) {
                std::swap((*this)(row1, j), (*this)(row2, j));
            }
        }

    public:
        // Default constructor for static matrices
        constexpr Matrix() noexcept requires(!isDynamic)
            : m_rows(static_cast<size_t>(Rows)),
              m_cols(static_cast<size_t>(Cols)),
              m_container{} {}

        // Variadic constructor for row vectors
        template<typename... InScalar>
            requires(Rows == 1 and Cols == DynamicSize and
                    (std::convertible_to<InScalar, Scalar> and ...))
        constexpr Matrix(const InScalar&... args) noexcept
            : m_rows(1),
              m_cols(sizeof...(InScalar)),
              m_container{ static_cast<Scalar>(args)... } {}

        // Default constructor for dynamic matrices
        constexpr Matrix() noexcept requires(isDynamic)
            : m_rows(isRowsDynamic ? 0 : static_cast<size_t>(Rows)),
              m_cols(isColsDynamic ? 0 : static_cast<size_t>(Cols)) {}

        // Size constructor for dynamic matrices
        explicit constexpr Matrix(size_t rows, size_t cols) requires(isDynamic)
            : m_rows(isRowsDynamic ? rows : static_cast<size_t>(Rows)),
              m_cols(isColsDynamic ? cols : static_cast<size_t>(Cols)),
              m_container(m_rows * m_cols) {}

        // Copy constructor (default)
        constexpr Matrix(const Matrix&) = default;
        
        // Move constructor (default - efficient for vector-based storage)
        constexpr Matrix(Matrix&&) noexcept = default;
        
        // Copy assignment (default)
        constexpr Matrix& operator=(const Matrix&) = default;
        
        // Move assignment (default)
        constexpr Matrix& operator=(Matrix&&) noexcept = default;

        // Initializer list constructor
        template<Numeric... InScalar>
            requires((std::convertible_to<InScalar, Scalar> and ...))
        constexpr Matrix(const std::initializer_list<InScalar>&... args)
            : m_rows(sizeof...(args)),
              m_cols(std::get<0>(std::tuple{ args... }).size()) {

            if constexpr (isDynamic) {
                m_container.resize(m_rows * m_cols);
            } else {
                assert(m_rows == static_cast<size_t>(Rows) and
                       m_cols == static_cast<size_t>(Cols));
            }
            
            Scalar* ptr = m_container.data();
            auto advance = [&](const auto& arg) {
                assert(arg.size() == m_cols);
                for (const auto& s : arg) {
                    *ptr++ = static_cast<Scalar>(s);
                }
            };
            (advance(args), ...);
        }

        // Copy constructor with type conversion
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar> and
                    CompatibleForAddition<Rows, Cols, InRows, InCols>)
        constexpr Matrix(const Matrix<InScalar, InRows, InCols>& rhs)
            : m_rows(rhs.rows()),
              m_cols(rhs.cols()) {
            
            if constexpr (isDynamic) {
                m_container.resize(m_rows * m_cols);
            } else {
                assert(m_rows == static_cast<size_t>(Rows) and
                       m_cols == static_cast<size_t>(Cols));
            }

            const size_t total = m_rows * m_cols;
            Scalar* dest = m_container.data();
            const InScalar* src = rhs.data();
            
            for (size_t i = 0; i < total; ++i) {
                dest[i] = static_cast<Scalar>(src[i]);
            }
        }
        
        // Move constructor with type conversion (only for exact dimension match)
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::same_as<Scalar, InScalar> and
                    std::same_as<std::integral_constant<ptrdiff_t, Rows>,
                                std::integral_constant<ptrdiff_t, InRows>> and
                    std::same_as<std::integral_constant<ptrdiff_t, Cols>,
                                std::integral_constant<ptrdiff_t, InCols>>)
        constexpr Matrix(Matrix<InScalar, InRows, InCols>&& rhs) noexcept
            : m_rows(rhs.rows()),
              m_cols(rhs.cols()),
              m_container(std::move(rhs.m_container)) {}

        // Static factory methods
        static constexpr Matrix Identity()
            requires(SquareMatrix<Rows, Cols>) {
            Matrix result;
            for (size_t i = 0; i < static_cast<size_t>(Rows); ++i) {
                result(i, i) = Scalar(1);
            }
            return result;
        }

        static constexpr Matrix Identity(size_t n)
            requires(DynamicMatrix<Rows, Cols>) {
            Matrix result(n, n);
            for (size_t i = 0; i < n; ++i) {
                result(i, i) = Scalar(1);
            }
            return result;
        }

        static constexpr Matrix Zeros()
            requires(StaticMatrix<Rows, Cols>) {
            return {};
        }
        
        static constexpr Matrix Zeros(size_t r, size_t c)
            requires(DynamicMatrix<Rows, Cols>) {
            return Matrix(r, c);
        }

        static constexpr Matrix Ones()
            requires(StaticMatrix<Rows, Cols>) {
            Matrix result;
            result.fill(Scalar(1));
            return result;
        }

        static constexpr Matrix Ones(size_t r, size_t c)
            requires(DynamicMatrix<Rows, Cols>) {
            Matrix result(r, c);
            result.fill(Scalar(1));
            return result;
        }

        static constexpr Matrix Constant(Scalar value)
            requires(StaticMatrix<Rows, Cols>) {
            Matrix result;
            result.fill(value);
            return result;
        }

        static constexpr Matrix Constant(size_t r, size_t c, Scalar value)
            requires(DynamicMatrix<Rows, Cols>) {
            Matrix result(r, c);
            result.fill(value);
            return result;
        }
        
        static Matrix Random(bool diagonal_dominance = false, bool symmetric = false, bool positive = false)
            requires(StaticMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

            Matrix result;

            if (positive) {
                // Generate positive definite matrix via A = B^T * B
                // This requires a square matrix
                static_assert(Rows == Cols, "Positive definite matrices must be square");

                Matrix B;
                for (size_t i = 0; i < static_cast<size_t>(Rows); ++i) {
                    for (size_t j = 0; j < static_cast<size_t>(Cols); ++j) {
                        B(i, j) = dist(gen);
                    }
                }

                // Compute A = B^T * B
                for (size_t i = 0; i < static_cast<size_t>(Rows); ++i) {
                    for (size_t j = 0; j < static_cast<size_t>(Cols); ++j) {
                        Scalar sum = Scalar(0);
                        for (size_t k = 0; k < static_cast<size_t>(Rows); ++k) {
                            sum += B(k, i) * B(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
            } else if (symmetric) {
                // Generate symmetric matrix
                static_assert(Rows == Cols, "Symmetric matrices must be square");

                for (size_t i = 0; i < static_cast<size_t>(Rows); ++i) {
                    for (size_t j = i; j < static_cast<size_t>(Cols); ++j) {
                        Scalar val = dist(gen);
                        result(i, j) = val;
                        result(j, i) = val;
                    }
                }
            } else {
                // Generate random matrix
                for (size_t i = 0; i < static_cast<size_t>(Rows); ++i) {
                    for (size_t j = 0; j < static_cast<size_t>(Cols); ++j) {
                        result(i, j) = dist(gen);
                    }
                }
            }

            if (diagonal_dominance and Rows == Cols) {
                // Make diagonally dominant
                for (size_t i = 0; i < static_cast<size_t>(Rows); ++i) {
                    Scalar row_sum = Scalar(0);
                    for (size_t j = 0; j < static_cast<size_t>(Cols); ++j) {
                        if (i != j) {
                            row_sum += std::abs(result(i, j));
                        }
                    }
                    result(i, i) = row_sum + Scalar(1); // Add 1 for stability
                }
            }

            return result;
        }

        static Matrix Random(size_t r, size_t c, bool diagonal_dominance = false, bool symmetric = false, bool positive = false)
            requires(DynamicMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

            Matrix result(r, c);

            if (positive) {
                // Generate positive definite matrix via A = B^T * B
                assert(r == c && "Positive definite matrices must be square");

                Matrix B(r, c);
                for (size_t i = 0; i < r; ++i) {
                    for (size_t j = 0; j < c; ++j) {
                        B(i, j) = dist(gen);
                    }
                }

                // Compute A = B^T * B
                for (size_t i = 0; i < r; ++i) {
                    for (size_t j = 0; j < c; ++j) {
                        Scalar sum = Scalar(0);
                        for (size_t k = 0; k < r; ++k) {
                            sum += B(k, i) * B(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
            } else if (symmetric) {
                // Generate symmetric matrix
                assert(r == c && "Symmetric matrices must be square");

                for (size_t i = 0; i < r; ++i) {
                    for (size_t j = i; j < c; ++j) {
                        Scalar val = dist(gen);
                        result(i, j) = val;
                        result(j, i) = val;
                    }
                }
            } else {
                // Generate random matrix
                for (size_t i = 0; i < r; ++i) {
                    for (size_t j = 0; j < c; ++j) {
                        result(i, j) = dist(gen);
                    }
                }
            }

            if (diagonal_dominance and r == c) {
                // Make diagonally dominant
                for (size_t i = 0; i < r; ++i) {
                    Scalar row_sum = Scalar(0);
                    for (size_t j = 0; j < c; ++j) {
                        if (i != j) {
                            row_sum += std::abs(result(i, j));
                        }
                    }
                    result(i, i) = row_sum + Scalar(1); // Add 1 for stability
                }
            }

            return result;
        }

        // Element access
        constexpr Scalar& operator()(size_t i, size_t j) noexcept {
            assert(i < m_rows and j < m_cols);
            return m_container[i * m_cols + j];
        }
        
        constexpr const Scalar& operator()(size_t i, size_t j) const noexcept {
            assert(i < m_rows and j < m_cols);
            return m_container[i * m_cols + j];
        }

        // Linear indexing for vectors
        constexpr Scalar& operator[](size_t i) noexcept
            requires(RowVector<Rows, Cols>) {
            assert(i < m_cols);
            return m_container[i];
        }
        
        constexpr const Scalar& operator[](size_t i) const noexcept
            requires(RowVector<Rows, Cols>) {
            assert(i < m_cols);
            return m_container[i];
        }

        // Dimensions
        constexpr size_t rows() const noexcept { return m_rows; }
        constexpr size_t cols() const noexcept { return m_cols; }
        constexpr size_t size() const noexcept { return m_rows * m_cols; }

        // Data access
        constexpr Scalar* data() noexcept { return m_container.data(); }
        constexpr const Scalar* data() const noexcept { return m_container.data(); }

        // Submatrix extraction
        constexpr Matrix<Scalar, DynamicSize, DynamicSize>
        submatrix(size_t rowOffset, size_t colOffset, size_t rows, size_t cols) const {
            assert(rowOffset + rows <= m_rows and colOffset + cols <= m_cols);
            
            Matrix<Scalar, DynamicSize, DynamicSize> result(rows, cols);
            
            if constexpr (std::is_trivially_copyable_v<Scalar>) {
                // Use memcpy for trivially copyable types
                for (size_t i = 0; i < rows; ++i) {
                    std::memcpy(
                        result.data() + i * cols,
                        data() + (rowOffset + i) * m_cols + colOffset,
                        cols * sizeof(Scalar)
                    );
                }
            } else {
                // Manual copy for non-trivial types
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        result(i, j) = (*this)(rowOffset + i, colOffset + j);
                    }
                }
            }
            
            return result;
        }

        // Fill with value
        constexpr void fill(Scalar value) noexcept {
            std::fill(m_container.begin(), m_container.end(), value);
        }

        // Dynamic matrix operations
        constexpr void resize(size_t r, size_t c)
            requires(DynamicMatrix<Rows, Cols>) {
            m_rows = isRowsDynamic ? r : m_rows;
            m_cols = isColsDynamic ? c : m_cols;
            m_container.resize(m_rows * m_cols);
        }

        constexpr void reserve(size_t capacity)
            requires(DynamicMatrix<Rows, Cols>) {
            m_container.reserve(capacity);
        }

        constexpr void clear()
            requires(DynamicMatrix<Rows, Cols>) {
            m_rows = 0;
            m_cols = 0;
            m_container.clear();
        }

        constexpr void shrink_to_fit()
            requires(DynamicMatrix<Rows, Cols>) {
            m_container.shrink_to_fit();
        }

        // Insert rows from another matrix at the end
        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(DynamicMatrix<Rows, Cols> and std::convertible_to<InScalar, Scalar>)
        constexpr void insert(const Matrix<InScalar, InRows, InCols>& other) {
            // If current matrix is empty, just copy the other matrix
            if (m_rows == 0 || m_cols == 0) {
                m_rows = other.rows();
                m_cols = other.cols();
                m_container.resize(m_rows * m_cols);
                for (size_t i = 0; i < m_rows * m_cols; ++i) {
                    m_container[i] = static_cast<Scalar>(other.data()[i]);
                }
                return;
            }

            // Otherwise, append rows (columns must match)
            assert(m_cols == other.cols());

            size_t old_rows = m_rows;
            size_t new_rows = m_rows + other.rows();

            // Resize container
            m_container.resize(new_rows * m_cols);
            m_rows = new_rows;

            // Copy new rows
            for (size_t i = 0; i < other.rows(); ++i) {
                for (size_t j = 0; j < m_cols; ++j) {
                    (*this)(old_rows + i, j) = static_cast<Scalar>(other(i, j));
                }
            }
        }

        // Swap
        constexpr void swap(Matrix& other) noexcept {
            std::swap(m_rows, other.m_rows);
            std::swap(m_cols, other.m_cols);
            std::swap(m_container, other.m_container);
        }

        // Dot product (for row vectors)
        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        constexpr auto dot(const Matrix<InScalar, InRows, InCols>& rhs) const {
            assert(m_rows == 1 and rhs.rows() == 1 and m_cols == rhs.cols());
            
            using ResultType = CommonScalar<InScalar>;
            ResultType result{};
            
            for (size_t i = 0; i < m_cols; ++i) {
                result += static_cast<ResultType>((*this)(0, i)) *
                          static_cast<ResultType>(rhs(0, i));
            }
            
            return result;
        }

        // Transpose (for fully dynamic matrices)
        constexpr Matrix<Scalar, DynamicSize, DynamicSize>
        transposed() const requires(isDynamic) {
            Matrix<Scalar, DynamicSize, DynamicSize> result(m_cols, m_rows);
            
            for (size_t i = 0; i < m_rows; ++i) {
                for (size_t j = 0; j < m_cols; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }
            
            return result;
        }
        
        // In-place transpose
        constexpr void transpose() requires(isRowsDynamic and isColsDynamic) {
            *this = transposed();
        }

        // Forward substitution: solve Lx = b where L is lower triangular with unit diagonal
        constexpr Matrix<Scalar, DynamicSize, DynamicSize>
        forward_substitution(const Matrix<Scalar, DynamicSize, DynamicSize>& b) const {
            assert(m_rows == m_cols);
            assert(b.rows() == m_rows);

            Matrix<Scalar, DynamicSize, DynamicSize> x(b.rows(), b.cols());

            for (size_t col = 0; col < b.cols(); ++col) {
                for (size_t i = 0; i < m_rows; ++i) {
                    Scalar sum = b(i, col);
                    for (size_t j = 0; j < i; ++j) {
                        sum -= (*this)(i, j) * x(j, col);
                    }
                    x(i, col) = sum; // L has unit diagonal
                }
            }

            return x;
        }

        // Backward substitution: solve Ux = b where U is upper triangular
        constexpr Matrix<Scalar, DynamicSize, DynamicSize>
        backward_substitution(const Matrix<Scalar, DynamicSize, DynamicSize>& b) const {
            assert(m_rows == m_cols);
            assert(b.rows() == m_rows);

            Matrix<Scalar, DynamicSize, DynamicSize> x(b.rows(), b.cols());

            for (size_t col = 0; col < b.cols(); ++col) {
                for (ptrdiff_t i = static_cast<ptrdiff_t>(m_rows) - 1; i >= 0; --i) {
                    Scalar sum = b(static_cast<size_t>(i), col);
                    for (size_t j = static_cast<size_t>(i) + 1; j < m_cols; ++j) {
                        sum -= (*this)(static_cast<size_t>(i), j) * x(j, col);
                    }
                    assert(std::abs((*this)(static_cast<size_t>(i), static_cast<size_t>(i))) >
                           std::numeric_limits<Scalar>::epsilon());
                    x(static_cast<size_t>(i), col) = sum / (*this)(static_cast<size_t>(i), static_cast<size_t>(i));
                }
            }

            return x;
        }

        // LU decomposition with partial pivoting
        // Returns tuple of (P, L, U) where PA = LU
        constexpr std::tuple<Matrix, Matrix, Matrix> lu() const
            requires(SquareMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            const size_t n = static_cast<size_t>(Rows);

            Matrix P = Identity();
            Matrix L = Zeros();
            Matrix U = *this;

            for (size_t k = 0; k < n; ++k) {
                // Find pivot row
                size_t pivot_row = k;
                Scalar max_val = std::abs(U(k, k));
                for (size_t i = k + 1; i < n; ++i) {
                    Scalar val = std::abs(U(i, k));
                    if (val > max_val) {
                        max_val = val;
                        pivot_row = i;
                    }
                }

                // Check for singular matrix
                assert(std::abs(U(pivot_row, k)) > std::numeric_limits<Scalar>::epsilon());

                // Swap rows if needed
                if (pivot_row != k) {
                    U.swap_rows(k, pivot_row);
                    P.swap_rows(k, pivot_row);
                    // Swap previously computed L elements
                    for (size_t j = 0; j < k; ++j) {
                        std::swap(L(k, j), L(pivot_row, j));
                    }
                }

                // Set diagonal of L to 1
                L(k, k) = Scalar(1);

                // Compute multipliers and eliminate below pivot
                for (size_t i = k + 1; i < n; ++i) {
                    L(i, k) = U(i, k) / U(k, k);
                    for (size_t j = k; j < n; ++j) {
                        U(i, j) -= L(i, k) * U(k, j);
                    }
                }
            }

            return std::make_tuple(P, L, U);
        }

        // LU decomposition for dynamic matrices
        constexpr std::tuple<Matrix<Scalar, DynamicSize, DynamicSize>,
                           Matrix<Scalar, DynamicSize, DynamicSize>,
                           Matrix<Scalar, DynamicSize, DynamicSize>> lu() const
            requires(DynamicMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            assert(m_rows == m_cols);
            const size_t n = m_rows;

            Matrix<Scalar, DynamicSize, DynamicSize> P = Matrix<Scalar, DynamicSize, DynamicSize>::Identity(n);
            Matrix<Scalar, DynamicSize, DynamicSize> L = Matrix<Scalar, DynamicSize, DynamicSize>::Zeros(n, n);
            Matrix<Scalar, DynamicSize, DynamicSize> U(*this);

            for (size_t k = 0; k < n; ++k) {
                // Find pivot row
                size_t pivot_row = k;
                Scalar max_val = std::abs(U(k, k));
                for (size_t i = k + 1; i < n; ++i) {
                    Scalar val = std::abs(U(i, k));
                    if (val > max_val) {
                        max_val = val;
                        pivot_row = i;
                    }
                }

                // Check for singular matrix
                assert(std::abs(U(pivot_row, k)) > std::numeric_limits<Scalar>::epsilon());

                // Swap rows if needed
                if (pivot_row != k) {
                    U.swap_rows(k, pivot_row);
                    P.swap_rows(k, pivot_row);
                    // Swap previously computed L elements
                    for (size_t j = 0; j < k; ++j) {
                        std::swap(L(k, j), L(pivot_row, j));
                    }
                }

                // Set diagonal of L to 1
                L(k, k) = Scalar(1);

                // Compute multipliers and eliminate below pivot
                for (size_t i = k + 1; i < n; ++i) {
                    L(i, k) = U(i, k) / U(k, k);
                    for (size_t j = k; j < n; ++j) {
                        U(i, j) -= L(i, k) * U(k, j);
                    }
                }
            }

            return std::make_tuple(P, L, U);
        }

        // Determinant using LU decomposition (static matrices)
        constexpr Scalar lu_det() const
            requires(SquareMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            auto [P, L, U] = lu();

            // Count row swaps in P to determine sign
            const size_t n = static_cast<size_t>(Rows);
            size_t swaps = 0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if (std::abs(P(i, j)) > std::numeric_limits<Scalar>::epsilon()) {
                        ++swaps;
                    }
                }
            }

            Scalar sign = (swaps % 2 == 0) ? Scalar(1) : Scalar(-1);

            // det(U) = product of diagonal elements
            Scalar det_U = Scalar(1);
            for (size_t i = 0; i < n; ++i) {
                det_U *= U(i, i);
            }

            return sign * det_U;
        }

        // Determinant using LU decomposition (dynamic matrices)
        constexpr Scalar lu_det() const
            requires(DynamicMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            assert(m_rows == m_cols);
            auto [P, L, U] = lu();

            // Count row swaps in P to determine sign
            const size_t n = m_rows;
            size_t swaps = 0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if (std::abs(P(i, j)) > std::numeric_limits<Scalar>::epsilon()) {
                        ++swaps;
                    }
                }
            }

            Scalar sign = (swaps % 2 == 0) ? Scalar(1) : Scalar(-1);

            // det(U) = product of diagonal elements
            Scalar det_U = Scalar(1);
            for (size_t i = 0; i < n; ++i) {
                det_U *= U(i, i);
            }

            return sign * det_U;
        }

        // Matrix inverse using LU decomposition (static matrices)
        constexpr Matrix lu_inv() const
            requires(SquareMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            auto [P, L, U] = lu();
            const size_t n = static_cast<size_t>(Rows);

            // Solve AX = I, which is PA = LU, so LUX = P
            // First compute P (permutation matrix is its own inverse when transposed)
            Matrix I = Identity();
            Matrix PI(I);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    PI(i, j) = Scalar(0);
                    for (size_t k = 0; k < n; ++k) {
                        PI(i, j) += P(i, k) * I(k, j);
                    }
                }
            }

            // Convert to dynamic for helper functions
            Matrix<Scalar, DynamicSize, DynamicSize> L_dyn(n, n);
            Matrix<Scalar, DynamicSize, DynamicSize> U_dyn(n, n);
            Matrix<Scalar, DynamicSize, DynamicSize> PI_dyn(n, n);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    L_dyn(i, j) = L(i, j);
                    U_dyn(i, j) = U(i, j);
                    PI_dyn(i, j) = PI(i, j);
                }
            }

            // Solve LY = PI
            auto Y = L_dyn.forward_substitution(PI_dyn);
            // Solve UX = Y
            auto X_dyn = U_dyn.backward_substitution(Y);

            // Convert back to static
            Matrix result;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result(i, j) = X_dyn(i, j);
                }
            }

            return result;
        }

        // Matrix inverse using LU decomposition (dynamic matrices)
        constexpr Matrix<Scalar, DynamicSize, DynamicSize> lu_inv() const
            requires(DynamicMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            assert(m_rows == m_cols);
            auto [P, L, U] = lu();
            const size_t n = m_rows;

            // Solve AX = I, which is PA = LU, so LUX = P
            auto I = Matrix<Scalar, DynamicSize, DynamicSize>::Identity(n);
            Matrix<Scalar, DynamicSize, DynamicSize> PI(n, n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    PI(i, j) = Scalar(0);
                    for (size_t k = 0; k < n; ++k) {
                        PI(i, j) += P(i, k) * I(k, j);
                    }
                }
            }

            // Solve LY = PI
            auto Y = L.forward_substitution(PI);
            // Solve UX = Y
            auto X = U.backward_substitution(Y);

            return X;
        }

        // Condition number using LU decomposition (static matrices)
        constexpr Scalar lu_cond() const
            requires(SquareMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            Scalar norm_A = infinity_norm();
            Matrix A_inv = lu_inv();
            Scalar norm_A_inv = A_inv.infinity_norm();
            return norm_A * norm_A_inv;
        }

        // Condition number using LU decomposition (dynamic matrices)
        constexpr Scalar lu_cond() const
            requires(DynamicMatrix<Rows, Cols> and FloatingPoint<Scalar>) {
            assert(m_rows == m_cols);
            Scalar norm_A = infinity_norm();
            auto A_inv = lu_inv();
            Scalar norm_A_inv = A_inv.infinity_norm();
            return norm_A * norm_A_inv;
        }

        // QR decomposition using Householder reflections
        constexpr std::pair<Matrix<Scalar, DynamicSize, DynamicSize>,
                          Matrix<Scalar, DynamicSize, DynamicSize>> qr() const
            requires(FloatingPoint<Scalar>) {
            const size_t m = m_rows;
            const size_t n = m_cols;
            const size_t min_mn = std::min(m, n);

            Matrix<Scalar, DynamicSize, DynamicSize> A_work(*this);
            Matrix<Scalar, DynamicSize, DynamicSize> Q = Matrix<Scalar, DynamicSize, DynamicSize>::Identity(m);

            for (size_t k = 0; k < min_mn; ++k) {
                // Extract column vector x = A[k:m, k]
                Matrix<Scalar, DynamicSize, DynamicSize> x(m - k, 1);
                for (size_t i = k; i < m; ++i) {
                    x(i - k, 0) = A_work(i, k);
                }

                // Compute norm of x
                Scalar x_norm = Scalar(0);
                for (size_t i = 0; i < m - k; ++i) {
                    x_norm += x(i, 0) * x(i, 0);
                }
                x_norm = std::sqrt(x_norm);

                if (x_norm < std::numeric_limits<Scalar>::epsilon()) {
                    continue; // Skip if column is already zero
                }

                // Create Householder vector v
                Matrix<Scalar, DynamicSize, DynamicSize> v = x;
                v(0, 0) += sign(x(0, 0)) * x_norm;

                // Normalize v
                Scalar v_norm = Scalar(0);
                for (size_t i = 0; i < m - k; ++i) {
                    v_norm += v(i, 0) * v(i, 0);
                }
                v_norm = std::sqrt(v_norm);

                if (v_norm > std::numeric_limits<Scalar>::epsilon()) {
                    for (size_t i = 0; i < m - k; ++i) {
                        v(i, 0) /= v_norm;
                    }

                    // Apply Householder reflection to A[k:m, k:n]
                    // A = A - 2 * v * (v^T * A)
                    for (size_t j = k; j < n; ++j) {
                        Scalar vt_A_j = Scalar(0);
                        for (size_t i = 0; i < m - k; ++i) {
                            vt_A_j += v(i, 0) * A_work(k + i, j);
                        }
                        for (size_t i = 0; i < m - k; ++i) {
                            A_work(k + i, j) -= Scalar(2) * v(i, 0) * vt_A_j;
                        }
                    }

                    // Update Q: Q = Q * H_k where H_k = I - 2*v*v^T
                    for (size_t j = 0; j < m; ++j) {
                        Scalar vt_Q_j = Scalar(0);
                        for (size_t i = 0; i < m - k; ++i) {
                            vt_Q_j += v(i, 0) * Q(k + i, j);
                        }
                        for (size_t i = 0; i < m - k; ++i) {
                            Q(k + i, j) -= Scalar(2) * v(i, 0) * vt_Q_j;
                        }
                    }
                }
            }

            // R is the upper triangular part of A_work
            Matrix<Scalar, DynamicSize, DynamicSize> R(min_mn, n);
            for (size_t i = 0; i < min_mn; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (j >= i) {
                        R(i, j) = A_work(i, j);
                    } else {
                        R(i, j) = Scalar(0);
                    }
                }
            }

            // Q should be m x min(m,n) for economy size
            Matrix<Scalar, DynamicSize, DynamicSize> Q_econ(m, min_mn);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < min_mn; ++j) {
                    Q_econ(i, j) = Q(i, j);
                }
            }

            return std::make_pair(Q_econ, R);
        }

        // Compute matrix rank using QR decomposition
        constexpr size_t rank() const
            requires(FloatingPoint<Scalar>) {
            if (m_rows == 0 || m_cols == 0) return 0;

            // Perform QR decomposition
            auto [Q, R] = qr();

            // Count non-zero diagonal elements in R
            size_t rank_count = 0;
            size_t min_dim = std::min(R.rows(), R.cols());

            for (size_t i = 0; i < min_dim; ++i) {
                if (std::abs(R(i, i)) > std::numeric_limits<Scalar>::epsilon()) {
                    ++rank_count;
                }
            }

            return rank_count;
        }

        // Unary negation
        constexpr Matrix operator-() const {
            Matrix result = *this;
            for (auto& elem : result.m_container) {
                elem = -elem;
            }
            return result;
        }

        // Compound assignment operators
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar> and
                    CompatibleForAddition<Rows, Cols, InRows, InCols>)
        constexpr Matrix& operator+=(const Matrix<InScalar, InRows, InCols>& rhs) {
            assert(m_rows == rhs.rows() and m_cols == rhs.cols());
            
            const size_t total = size();
            Scalar* lhs_data = data();
            const InScalar* rhs_data = rhs.data();
            
            for (size_t i = 0; i < total; ++i) {
                lhs_data[i] += static_cast<Scalar>(rhs_data[i]);
            }
            
            return *this;
        }

        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar> and
                    CompatibleForAddition<Rows, Cols, InRows, InCols>)
        constexpr Matrix& operator-=(const Matrix<InScalar, InRows, InCols>& rhs) {
            assert(m_rows == rhs.rows() and m_cols == rhs.cols());
            
            const size_t total = size();
            Scalar* lhs_data = data();
            const InScalar* rhs_data = rhs.data();
            
            for (size_t i = 0; i < total; ++i) {
                lhs_data[i] -= static_cast<Scalar>(rhs_data[i]);
            }
            
            return *this;
        }

        constexpr Matrix& operator*=(Scalar rhs) noexcept {
            for (auto& elem : m_container) {
                elem *= rhs;
            }
            return *this;
        }

        constexpr Matrix& operator/=(Scalar rhs) {
            assert(std::abs(rhs) > std::numeric_limits<Scalar>::epsilon());
            
            for (auto& elem : m_container) {
                elem /= rhs;
            }
            return *this;
        }

        // Comparison operators
        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        constexpr bool operator==(const Matrix<InScalar, InRows, InCols>& rhs) const noexcept {
            if (m_rows != rhs.rows() or m_cols != rhs.cols()) {
                return false;
            }
            
            return std::equal(m_container.begin(), m_container.end(), rhs.data());
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        constexpr bool operator!=(const Matrix<InScalar, InRows, InCols>& rhs) const noexcept {
            return !(*this == rhs);
        }

        // Element-wise (Hadamard) multiplication
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        constexpr auto cwiseProduct(const Matrix<InScalar, InRows, InCols>& rhs) const {
            assert(m_rows == rhs.rows() and m_cols == rhs.cols());
            
            using ResultScalar = CommonScalar<InScalar>;
            Matrix<ResultScalar, Rows, Cols> result(m_rows, m_cols);
            
            for (size_t i = 0; i < size(); ++i) {
                result.data()[i] = static_cast<ResultScalar>(data()[i]) *
                                   static_cast<ResultScalar>(rhs.data()[i]);
            }
            
            return result;
        }

        // Element-wise division
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        constexpr auto cwiseQuotient(const Matrix<InScalar, InRows, InCols>& rhs) const {
            assert(m_rows == rhs.rows() and m_cols == rhs.cols());
            
            using ResultScalar = CommonScalar<InScalar>;
            Matrix<ResultScalar, Rows, Cols> result(m_rows, m_cols);
            
            for (size_t i = 0; i < size(); ++i) {
                assert(std::abs(rhs.data()[i]) > std::numeric_limits<InScalar>::epsilon());
                result.data()[i] = static_cast<ResultScalar>(data()[i]) /
                                   static_cast<ResultScalar>(rhs.data()[i]);
            }
            
            return result;
        }

        // Trace (for square matrices)
        constexpr Scalar trace() const
            requires(SquareMatrix<Rows, Cols>) {
            Scalar sum{};
            for (size_t i = 0; i < static_cast<size_t>(Rows); ++i) {
                sum += (*this)(i, i);
            }
            return sum;
        }

        constexpr Scalar trace() const
            requires(DynamicMatrix<Rows, Cols>) {
            assert(m_rows == m_cols);
            Scalar sum{};
            for (size_t i = 0; i < m_rows; ++i) {
                sum += (*this)(i, i);
            }
            return sum;
        }

        // Norm (for vectors)
        constexpr Scalar norm() const
            requires(RowVector<Rows, Cols> and FloatingPoint<Scalar>) {
            Scalar sum{};
            for (size_t i = 0; i < m_cols; ++i) {
                sum += (*this)(0, i) * (*this)(0, i);
            }
            return std::sqrt(sum);
        }

        // Normalize (for vectors)
        constexpr Matrix& normalize()
            requires(RowVector<Rows, Cols> and FloatingPoint<Scalar>) {
            Scalar n = norm();
            assert(n > std::numeric_limits<Scalar>::epsilon());
            *this /= n;
            return *this;
        }

        constexpr Matrix normalized() const
            requires(RowVector<Rows, Cols> and FloatingPoint<Scalar>) {
            Matrix result = *this;
            result.normalize();
            return result;
        }

        // Binary arithmetic operators
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        friend constexpr auto operator+(
            const Matrix& lhs,
            const Matrix<InScalar, InRows, InCols>& rhs
        ) {
            using ResultScalar = CommonScalar<InScalar>;
            Matrix<ResultScalar, Rows, Cols> result(lhs);
            result += rhs;
            return result;
        }
        
        // Addition: rvalue + lvalue (reuse lhs storage)
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        friend constexpr auto operator+(
            Matrix&& lhs,
            const Matrix<InScalar, InRows, InCols>& rhs
        ) {
            lhs += rhs;
            return std::move(lhs);
        }
        
        // Addition: lvalue + rvalue (reuse rhs storage if same type)
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols> and
                    std::same_as<Scalar, InScalar>)
        friend constexpr auto operator+(
            const Matrix& lhs,
            Matrix<InScalar, InRows, InCols>&& rhs
        ) {
            rhs += lhs;
            return std::move(rhs);
        }
        
        // Addition: rvalue + rvalue (reuse lhs storage)
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        friend constexpr auto operator+(
            Matrix&& lhs,
            Matrix<InScalar, InRows, InCols>&& rhs
        ) {
            lhs += rhs;
            return std::move(lhs);
        }

        // Subtraction: lvalue - lvalue
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        friend constexpr auto operator-(
            const Matrix& lhs,
            const Matrix<InScalar, InRows, InCols>& rhs
        ) {
            using ResultScalar = CommonScalar<InScalar>;
            Matrix<ResultScalar, Rows, Cols> result(lhs);
            result -= rhs;
            return result;
        }
        
        // Subtraction: rvalue - lvalue (reuse lhs storage)
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        friend constexpr auto operator-(
            Matrix&& lhs,
            const Matrix<InScalar, InRows, InCols>& rhs
        ) {
            lhs -= rhs;
            return std::move(lhs);
        }
        
        // Subtraction: lvalue - rvalue (cannot reuse rhs easily)
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        friend constexpr auto operator-(
            const Matrix& lhs,
            Matrix<InScalar, InRows, InCols>&& rhs
        ) {
            using ResultScalar = CommonScalar<InScalar>;
            Matrix<ResultScalar, Rows, Cols> result(lhs);
            result -= rhs;
            return result;
        }
        
        // Subtraction: rvalue - rvalue (reuse lhs storage)
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForAddition<Rows, Cols, InRows, InCols>)
        friend constexpr auto operator-(
            Matrix&& lhs,
            Matrix<InScalar, InRows, InCols>&& rhs
        ) {
            lhs -= rhs;
            return std::move(lhs);
        }

        // Optimized matrix multiplication
        template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(CompatibleForMultiplication<Cols, InRows>)
        friend constexpr auto operator*(
            const Matrix& lhs,
            const Matrix<InScalar, InRows, InCols>& rhs
        ) {
            assert(lhs.cols() == rhs.rows());
            
            using ResultScalar = CommonScalar<InScalar>;
            using ResultMatrix = Matrix<ResultScalar,
                (Rows == DynamicSize ? DynamicSize : Rows),
                (InCols == DynamicSize ? DynamicSize : InCols)>;
            
            ResultMatrix result(lhs.rows(), rhs.cols());
            result.fill(ResultScalar(0));
            
            // Cache-friendly implementation: i-k-j order
            for (size_t i = 0; i < lhs.rows(); ++i) {
                for (size_t k = 0; k < lhs.cols(); ++k) {
                    const ResultScalar lhs_ik = static_cast<ResultScalar>(lhs(i, k));
                    for (size_t j = 0; j < rhs.cols(); ++j) {
                        result(i, j) += lhs_ik * static_cast<ResultScalar>(rhs(k, j));
                    }
                }
            }
            
            return result;
        }

        // Scalar multiplication: matrix * scalar (lvalue)
        template<Numeric InScalar>
            requires(std::convertible_to<InScalar, Scalar>)
        friend constexpr Matrix operator*(const Matrix& lhs, InScalar rhs) {
            Matrix result = lhs;
            result *= static_cast<Scalar>(rhs);
            return result;
        }
        
        // Scalar multiplication: matrix * scalar (rvalue - reuse storage)
        template<Numeric InScalar>
            requires(std::convertible_to<InScalar, Scalar>)
        friend constexpr Matrix operator*(Matrix&& lhs, InScalar rhs) {
            lhs *= static_cast<Scalar>(rhs);
            return std::move(lhs);
        }

        // Scalar multiplication: scalar * matrix (lvalue)
        template<Numeric InScalar>
            requires(std::convertible_to<InScalar, Scalar>)
        friend constexpr Matrix operator*(InScalar lhs, const Matrix& rhs) {
            return rhs * lhs;
        }
        
        // Scalar multiplication: scalar * matrix (rvalue)
        template<Numeric InScalar>
            requires(std::convertible_to<InScalar, Scalar>)
        friend constexpr Matrix operator*(InScalar lhs, Matrix&& rhs) {
            return std::move(rhs) * lhs;
        }

        // Scalar division: matrix / scalar (lvalue)
        template<Numeric InScalar>
            requires(std::convertible_to<InScalar, Scalar>)
        friend constexpr Matrix operator/(const Matrix& lhs, InScalar rhs) {
            assert(std::abs(rhs) > std::numeric_limits<Scalar>::epsilon());
            Matrix result = lhs;
            result /= static_cast<Scalar>(rhs);
            return result;
        }
        
        // Scalar division: matrix / scalar (rvalue - reuse storage)
        template<Numeric InScalar>
            requires(std::convertible_to<InScalar, Scalar>)
        friend constexpr Matrix operator/(Matrix&& lhs, InScalar rhs) {
            assert(std::abs(rhs) > std::numeric_limits<Scalar>::epsilon());
            lhs /= static_cast<Scalar>(rhs);
            return std::move(lhs);
        }
    };

    // Type aliases
    template<typename Type, ptrdiff_t Size>
    using MatrixX = Matrix<Type, Size, Size>;

    using Matrix2d = MatrixX<double, 2>;
    using Matrix3d = MatrixX<double, 3>;
    using Matrix4d = MatrixX<double, 4>;
    using Matrix5d = MatrixX<double, 5>;

    template<typename Type, ptrdiff_t Size = DynamicSize>
    using Vector = Matrix<Type, 1, Size>;

    template<typename Type, ptrdiff_t Size>
    using VectorX = Matrix<Type, 1, Size>;

    using Vector2d = VectorX<double, 2>;
    using Vector3d = VectorX<double, 3>;
    using Vector4d = VectorX<double, 4>;

    // Stream output operator
    template<Numeric InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
    std::ostream& operator<<(
        std::ostream& os,
        const Matrix<InScalar, InRows, InCols>& matrix
    ) {
        for (size_t i = 0; i < matrix.rows(); ++i) {
            os << '(';
            for (size_t j = 0; j < matrix.cols(); ++j) {
                os << matrix(i, j);
                if (j + 1 < matrix.cols()) {
                    os << ' ';
                }
            }
            os << ')';
            if (i + 1 < matrix.rows()) {
                os << '\n';
            }
        }
        return os;
    }
    
    // Swap specialization for std::swap
    template<Numeric Scalar, ptrdiff_t Rows, ptrdiff_t Cols>
    constexpr void swap(Matrix<Scalar, Rows, Cols>& lhs,
                       Matrix<Scalar, Rows, Cols>& rhs) noexcept {
        lhs.swap(rhs);
    }

} // namespace linear

#endif // MATRIX_HPP



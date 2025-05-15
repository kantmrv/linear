#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <omp.h>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <vector>


namespace linear {
    constexpr ptrdiff_t DynamicSize = -1;

    template<typename Scalar_, ptrdiff_t Rows_ = DynamicSize, ptrdiff_t Cols_ = DynamicSize>
    class Matrix {
    public:
        using Scalar = Scalar_;
        inline static constexpr ptrdiff_t Rows = Rows_;
        inline static constexpr ptrdiff_t Cols = Cols_;

        inline static constexpr bool isRowsDynamic = (Rows == DynamicSize);
        inline static constexpr bool isColsDynamic = (Cols == DynamicSize);
        inline static constexpr bool isDynamic = (isRowsDynamic || isColsDynamic);
    private:
        template<typename InScalar>
        using CommonScalar = std::common_type_t<Scalar, InScalar>;
        template<typename InScalar>
        using CommonMatrix = Matrix<CommonScalar<InScalar>, Rows, Cols>;
        using StaticContainer = std::array<Scalar, (isDynamic ? 1 : Rows * Cols)>;
        using DynamicContainer = std::vector<Scalar>;
        using ContainerType = std::conditional_t<isDynamic, DynamicContainer, StaticContainer>;

        size_t        m_rows;
        size_t        m_cols;
        ContainerType m_container;

    public:
        // Constructors
        constexpr Matrix() noexcept requires(!isDynamic)
            : m_rows(static_cast<size_t>(Rows)),
            m_cols(static_cast<size_t>(Cols)),
            m_container{} {}

        template<typename... InScalar>
            requires(Rows == 1 && (std::convertible_to<InScalar, Scalar> && ...))
        constexpr Matrix(const InScalar&... args) noexcept
            : m_rows(1),
            m_cols(static_cast<size_t>(sizeof...(InScalar))),
            m_container{ static_cast<Scalar>(args)... } {}

        constexpr Matrix() noexcept requires(isDynamic)
            : m_rows(isRowsDynamic ? 0 : static_cast<size_t>(Rows)),
            m_cols(isColsDynamic ? 0 : static_cast<size_t>(Cols)) {}

        explicit constexpr Matrix(size_t rows, size_t cols) requires(isDynamic)
            : m_rows(isRowsDynamic ? rows : static_cast<size_t>(Rows)),
            m_cols(isColsDynamic ? cols : static_cast<size_t>(Cols)) {
            m_container.resize(m_rows * m_cols);
        }

        template<typename... InScalar>
            requires(std::convertible_to<InScalar, Scalar> && ...)
        constexpr Matrix(const std::initializer_list<InScalar>&... args)
            : m_rows(sizeof...(args)),
            m_cols(std::get<0>(std::tuple{ args... }).size()) {


            if constexpr (isDynamic) {
                m_container.resize(m_rows * m_cols);
            } else {
                assert(m_rows == Rows && m_cols == Cols);
            }
            Scalar* ptr = m_container.data();
            auto advance = [&](auto const& arg) {
                assert(arg.size() == m_cols);
#pragma omp simd
                for (auto const& s : arg) {
                    *ptr++ = static_cast<Scalar>(s);
                }
            };
            (advance(args), ...);
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        constexpr Matrix(const Matrix<InScalar, InRows, InCols>& rhs) noexcept
            : m_rows(rhs.rows()),
            m_cols(rhs.cols()) {
            if constexpr (isDynamic) {
                m_container.resize(m_rows * m_cols);
            } else {
                assert(m_rows == Rows && m_cols == Cols);
            }
#pragma omp simd
            for (size_t i = 0; i < m_rows * m_cols; ++i)
                m_container.data()[i] = static_cast<Scalar>(rhs.data()[i]);
        }

        // Static factories
        static constexpr Matrix Identity() requires(!isDynamic && Rows == Cols) {
            Matrix lhs;
#pragma omp simd
            for (size_t i = 0; i < Rows; ++i) lhs(i, i) = Scalar(1);
            return lhs;
        }

        constexpr static Matrix Identity(size_t n) requires(isDynamic) {
            Matrix lhs(n, n);
#pragma omp simd
            for (size_t i = 0; i < n; ++i) lhs(i, i) = Scalar(1);
            return lhs;
        }

        static constexpr Matrix Zeros() requires(!isDynamic) { return {}; }
        constexpr static Matrix Zeros(size_t r, size_t c) requires(isDynamic) { return Matrix(r, c); }

        // Element access
        constexpr Scalar& operator()(size_t i, size_t j) noexcept {
            assert(i < m_rows && j < m_cols);
            return m_container[i * m_cols + j];
        }
        constexpr const Scalar& operator()(size_t i, size_t j) const noexcept {
            assert(i < m_rows && j < m_cols);
            return m_container[i * m_cols + j];
        }

        // Dimensions
        constexpr size_t rows() const noexcept { return m_rows; }
        constexpr size_t cols() const noexcept { return m_cols; }
        constexpr size_t size() const noexcept { return m_rows * m_cols; }

        // Data pointer
        constexpr Scalar* data() noexcept { return m_container.data(); }
        constexpr const Scalar* data() const noexcept { return m_container.data(); }

        // Submatrix
        constexpr Matrix<Scalar, DynamicSize, DynamicSize>
            submatrix(
                size_t rowOffset = 0,
                size_t colOffset = 0,
                size_t rows = 1,
                size_t cols = 1
            ) const noexcept {
            assert(rowOffset + rows <= m_rows && colOffset + cols <= m_cols);
            Matrix<Scalar, DynamicSize, DynamicSize> out(rows, cols);
#pragma omp simd
            for (size_t i = 0; i < rows; ++i) {
                std::memcpy(
                    out.data() + i * cols,
                    data() + (rowOffset + i) * m_cols + colOffset,
                    cols * sizeof(Scalar)
                );
            }
            return out;
        }

        constexpr size_t rank() const noexcept {
            using Type = std::conditional_t<std::is_integral_v<Scalar>, double, Scalar>;
            Matrix<Type, Rows, Cols> copy = *this;

            auto abs_c = [](Type x) constexpr { return x < Type(0) ? -x : x; };
            auto max_c = [](size_t a, size_t b) constexpr { return a > b ? a : b; };

            // 1) find global_max
            Type global_max = Type(0);

            for (size_t i = 0, N = m_rows * m_cols; i < N; ++i) {
                Type v = abs_c(copy.data()[i]);
                if (v > global_max) global_max = v;
            }
            if (global_max == Type(0)) return 0;

            // 2) compute eps
            Type eps;
            if constexpr (std::is_floating_point_v<Type>) {
                eps = std::numeric_limits<Type>::epsilon()
                    * Type(max_c(m_rows, m_cols))
                    * global_max;
            } else {
                eps = Type(std::numeric_limits<double>::epsilon())
                    * Type(max_c(m_rows, m_cols))
                    * global_max;
            }

            size_t rank = 0;

            // 3) main Gaussian‐elimination loop
            for (size_t c = 0; c < m_cols && rank < m_rows; ++c) {
                // 3a) find pivot
                size_t piv = rank;
                Type pivot_val = abs_c(copy(rank, c));

                for (size_t i = rank + 1; i < m_rows; ++i) {
                    Type v = abs_c(copy(i, c));
                    if (v > pivot_val) {
                        pivot_val = v;
                        piv = i;
                    }
                }
                if (pivot_val <= eps) continue;

                // 3b) swap rows if needed
                if (piv != rank) {
                    for (size_t j = 0; j < m_cols; ++j) {
                        Type tmp = copy(rank, j);
                        copy(rank, j) = copy(piv, j);
                        copy(piv, j) = tmp;
                    }
                }

                // 3c) eliminate below
                Type* base = copy.data() + rank * m_cols;
                for (size_t i = rank + 1; i < m_rows; ++i) {
                    Type* row = copy.data() + i * m_cols;
                    Type  fac = row[c] / base[c];
#pragma omp simd
                    for (size_t j = c; j < m_cols; ++j) {
                        row[j] -= fac * base[j];
                    }
                }

                ++rank;
            }

            return rank;
        }


        // Fill
        constexpr void fill(Scalar v) noexcept {
            for (auto& e : m_container) e = v;
        }

        // Insert (dynamic only)
        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        constexpr void insert(const Matrix<InScalar, InRows, InCols>& rhs) noexcept requires(isRowsDynamic) {
            if (isColsDynamic && m_cols == 0) {
                m_cols = rhs.cols();
            } else{
                assert(cols() == rhs.cols());
            }
            size_t old = m_rows;
            m_rows = m_rows + rhs.rows();
            m_container.resize(m_rows * m_cols);
            std::memcpy(
                    m_container.data() + old * m_cols,
                    rhs.data(),
                    rhs.rows() * m_cols * sizeof(Scalar)
            );
        }

        // Resize (dynamic only)
        constexpr void resize(size_t r, size_t c) noexcept requires(isDynamic) {
            m_rows = r; m_cols = c;
            m_container.resize(r * c);
        }

        constexpr void shrink_to_fit() noexcept requires(isDynamic) {
            m_container.shrink_to_fit();
        }

        // Dot product of row vectors
        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        constexpr Scalar dot(const Matrix<InScalar, InRows, InCols>& rhs) const {
            assert(rows() == 1 && rhs.rows() == 1 && cols() == rhs.cols());
            Scalar out{};
#pragma omp simd
            for (size_t i = 0; i < cols(); ++i) out += (*this)(0, i) * rhs(0, i);
            return out;
        }

        // Transpose (dynamic only)
        constexpr void transpose() noexcept requires(isRowsDynamic&& isColsDynamic) {
            Matrix out(cols(), rows());
            for (size_t i = 0; i < rows(); ++i)
#pragma omp simd
                for (size_t j = 0; j < cols(); ++j) out(j, i) = (*this)(i, j);
            *this = std::move(out);
        }

        // Unary negation
        constexpr Matrix operator-() const noexcept {
            Matrix out = *this;
#pragma omp simd
            for (auto& e : out.m_container) e = -e;
            return out;
        }

        // Compound assignment
        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        constexpr CommonMatrix<InScalar>& operator+=(const Matrix<InScalar, InRows, InCols>& rhs) noexcept {
            assert(rows() == rhs.rows() && cols() == rhs.cols());
#pragma omp simd
            for (size_t i = 0; i < size(); ++i) m_container.data()[i] += rhs.data()[i];
            return *reinterpret_cast<CommonMatrix<InScalar>*>(this);
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        constexpr CommonMatrix<InScalar>& operator-=(const Matrix<InScalar, InRows, InCols>& rhs) noexcept {
            assert(rows() == rhs.rows() && cols() == rhs.cols());
#pragma omp simd
            for (size_t i = 0; i < size(); ++i) m_container.data()[i] -= rhs.data()[i];
            return *reinterpret_cast<CommonMatrix<InScalar>*>(this);
        }

        constexpr Matrix& operator*=(Scalar rhs) noexcept {
            for (auto& e : m_container) e *= rhs;
            return *this;
        }

        constexpr Matrix& operator/=(Scalar rhs) noexcept {
            assert(rhs > Scalar(std::numeric_limits<double>::epsilon()));
            for (auto& e : m_container) e /= rhs;
            return *this;
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        constexpr bool operator==(const Matrix<InScalar, InRows, InCols>& rhs) const noexcept {
            return m_rows == rhs.rows() && m_cols == rhs.cols() && std::equal(m_container.begin(), m_container.end(), rhs.data());;
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        constexpr bool operator!=(const Matrix<InScalar, InRows, InCols>& rhs) const noexcept {
            return !(*this == rhs);
        }


        // Free operators
        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        friend constexpr auto operator+(const Matrix& lhs, const Matrix<InScalar, InRows, InCols>& rhs) noexcept {
            auto out = static_cast<typename Matrix::template CommonMatrix<InScalar>>(lhs);
            return out += rhs;
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        friend constexpr auto operator-(const Matrix& lhs, const Matrix<InScalar, InRows, InCols>& rhs) noexcept {
            auto out = static_cast<typename Matrix::template CommonMatrix<InScalar>>(lhs);
            return out -= rhs;
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
        friend constexpr Matrix<typename Matrix::template CommonScalar<InScalar>> operator*(
            const Matrix& lhs, const Matrix<InScalar, InRows, InCols>& rhs) noexcept {
            assert(lhs.cols() == rhs.rows());
            using Out = Matrix<typename Matrix::template CommonScalar<InScalar>>;
            Out out(lhs.rows(), rhs.cols());
            out.fill(Out::Scalar(0));
            for (size_t i = 0; i < lhs.rows(); ++i) {
                for (size_t k = 0; k < lhs.cols(); ++k) {
                    auto s = static_cast<typename Out::Scalar>(lhs(i, k));
#pragma omp simd
                    for (size_t j = 0; j < rhs.cols(); ++j)
                        out(i, j) += s * rhs(k, j);
                }
            }
            return out;
        }

        template<typename InScalar>
        friend constexpr Matrix operator*(const Matrix& lhs, InScalar rhs) noexcept
            requires(std::convertible_to<InScalar, Scalar>) {
            auto out = lhs;
            out *= static_cast<Scalar>(rhs);
            return out;
        }

        template<typename InScalar>
        friend constexpr Matrix operator*(InScalar lhs, const Matrix& rhs) noexcept
            requires(std::convertible_to<InScalar, Scalar>) {
            return rhs * lhs;
        }

        template<typename InScalar>
        friend constexpr Matrix operator/(const Matrix& lhs, InScalar rhs) noexcept
            requires(std::convertible_to<InScalar, Scalar>) {
            assert(rhs > static_cast<InScalar>(std::numeric_limits<double>::epsilon()));
            auto out = lhs;
            out /= static_cast<Scalar>(rhs);
            return out;
        }
    };

    // Matrix alias
    template<typename Type, ptrdiff_t Size>
    using MatrixX = Matrix<Type, Size, Size>;

    using Matrix2d = MatrixX<double, 2>;
    using Matrix3d = MatrixX<double, 3>;
    using Matrix4d = MatrixX<double, 4>;
    using Matrix5d = MatrixX<double, 5>;

    // Vector alias
    template<typename Type, ptrdiff_t Size = DynamicSize>
    using Vector = Matrix<Type, 1, Size>;

    template<typename Type, ptrdiff_t Size>
    using VectorX = Matrix<Type, 1, Size>;

    using Vector2d = VectorX<double, 2>;
    using Vector3d = VectorX<double, 3>;
    using Vector4d = VectorX<double, 4>;

    template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
    std::ostream& operator<<(std::ostream& lhs, const Matrix<InScalar, InRows, InCols>& rhs) {
        for (size_t i = 0; i < rhs.rows(); ++i) {
            lhs << '(';
            for (size_t j = 0; j < rhs.cols(); ++j) {
                lhs << rhs(i, j);
                if (j + 1 < rhs.cols()) lhs << ' ';
            }
            lhs << ')';
            if (i + 1 < rhs.rows()) lhs << '\n';
        }
        return lhs;
    }
} // namespace linear

#endif // MATRIX_H
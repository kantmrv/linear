//
//  src/Span.hpp
//  linear
//
//  Created by Alexandr on 01.10.2025.
//

#ifndef LINEAR_SPAN_HPP
#define LINEAR_SPAN_HPP

#include <random>

#include "Matrix.hpp"

namespace linear {

    // Span class represents a vector space span
    template <typename Scalar_ = long double>
        requires(std::is_floating_point_v<Scalar_>)
    class Span {
      private:
        using Scalar = Scalar_;
        Matrix<Scalar, DynamicSize, DynamicSize> m_basis;

        // Check if a row vector is linearly independent from the current basis
        template <typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        bool is_linearly_independent(const Matrix<InScalar, InRows, InCols>& row) const {
            if (row.rows() != 1) {
                throw std::invalid_argument("is_linearly_independent expects a row vector");
            }
            if (m_basis.rows() == 0) {
                return true;
            }
            Matrix<Scalar, DynamicSize, DynamicSize> ext = m_basis;
            Matrix<Scalar, DynamicSize, DynamicSize> row_mat(1, row.cols());
            for (size_t j = 0; j < row.cols(); ++j) {
                row_mat(0, j) = static_cast<Scalar>(row(0, j));
            }
            ext.insert(row_mat);
            return ext.qr_rank() > m_basis.qr_rank();
        }

      public:
        // Default constructor creates an empty span
        constexpr Span() noexcept
            : m_basis(0, 0) {}

        // Construct span from matrix rows (extracts linearly independent rows)
        template <typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        explicit Span(const Matrix<InScalar, InRows, InCols>& mat)
            : m_basis(0, mat.cols()) {
            for (size_t i = 0; i < mat.rows(); ++i) {
                Matrix<Scalar> row(1, mat.cols()); // Create 1×n matrix (row vector)
                for (size_t j = 0; j < mat.cols(); ++j) {
                    row(0, j) = static_cast<Scalar>(mat(i, j));
                }
                if (is_linearly_independent(row)) {
                    insert(row);
                }
            }
        }

        template <typename... InScalar, ptrdiff_t Size>
            requires(std::conjunction_v<std::is_convertible<InScalar, Vector<Scalar>>...>)
        Span(const Vector<InScalar, Size>&... vs)
            : m_basis(0, (sizeof...(InScalar) ? std::get<0>(std::tuple{vs...}).size() : 0)) {
            (insert(vs), ...);
        }

        // Insert a vector into the span if it's linearly independent
        void insert(const Vector<Scalar>& v) {
            if (v.qr_rank() == 0) {
                return;
            }
            if (is_linearly_independent(v)) {
                Matrix<Scalar, DynamicSize, DynamicSize> row(1, v.size());
                for (size_t j = 0; j < v.size(); ++j) {
                    row(0, j) = v(0, j);
                }
                m_basis.insert(row);
            }
        }

        // Insert a matrix (row vector) into the span if it's linearly independent
        template <typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        void insert(const Matrix<InScalar, InRows, InCols>& v) {
            if (v.rows() != 1) {
                throw std::invalid_argument("insert expects a row vector (1×n matrix)");
            }
            if (v.qr_rank() == 0) {
                return;
            }
            if (is_linearly_independent(v)) {
                Matrix<Scalar, DynamicSize, DynamicSize> row(1, v.cols());
                for (size_t j = 0; j < v.cols(); ++j) {
                    row(0, j) = static_cast<Scalar>(v(0, j));
                }
                m_basis.insert(row);
            }
        }

        // Check if a vector is contained in the span
        template <typename InScalar, ptrdiff_t Size>
        constexpr bool contains(const Vector<InScalar, Size>& v) const {
            if (v.size() != m_basis.cols()) {
                return false;
            }
            if (v.qr_rank() == 0) {
                return true;
            }
            Matrix<Scalar, DynamicSize, DynamicSize> ext = m_basis;
            ext.insert(v);
            return ext.qr_rank() == m_basis.qr_rank();
        }

        // Check if all rows of a matrix are contained in the span
        template <typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        constexpr bool contains(const Matrix<InScalar, InRows, InCols>& m) const {
            if (m.cols() != m_basis.cols()) {
                return false;
            }
            if (m.qr_rank() == 0) {
                return true;
            }
            Matrix<Scalar, DynamicSize, DynamicSize> ext = m_basis;
            ext.insert(m);
            return ext.qr_rank() == m_basis.qr_rank();
        }

        // Return the dimension (number of basis vectors) of the span
        constexpr size_t dimension() const noexcept { return m_basis.rows(); }

        // Get the basis matrix
        constexpr auto& basis() const noexcept { return m_basis; }

        // Compute orthonormal basis using QR decomposition
        Matrix<Scalar, DynamicSize, DynamicSize> orthonormal_basis() const {
            if (m_basis.rows() == 0 || m_basis.cols() == 0) {
                return Matrix<Scalar, DynamicSize, DynamicSize>(0, 0);
            }

            // Transpose basis (rows → columns) for QR decomposition
            auto basis_T = m_basis.transposed();

            // QR decomposition produces Q with orthonormal columns
            auto [Q, R] = basis_T.qr();

            // Transpose Q back to get orthonormal rows matching original basis
            // structure
            return Q.transposed();
        }

        // Generate a random vector in the span by taking a random linear combination
        // of basis vectors
        Vector<Scalar> random() const {
            size_t r = m_basis.rows();
            size_t c = m_basis.cols();
            Matrix<Scalar> out(1, c);
            if (!r || !c) {
                return out;
            }

            std::mt19937_64 rng(std::random_device{}());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            // Generate random coefficients for each basis vector
            for (size_t i = 0; i < r; ++i) {
                Scalar coef = static_cast<Scalar>(dist(rng));
                // Add coef * basis[i] to the result
                for (size_t j = 0; j < c; ++j) {
                    out(0, j) += coef * m_basis(i, j);
                }
            }
            return out;
        }
    };

} // namespace linear

#endif // LINEAR_SPAN_HPP

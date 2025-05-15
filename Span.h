#ifndef LINEAR_SPAN_H
#define LINEAR_SPAN_H

#include <cmath>
#include <limits>
#include <random>

#include "Matrix.h"


namespace linear {

    template<typename Scalar_ = long double>
        requires(std::is_floating_point_v<Scalar_>)
    class Span {
    private:
        using Scalar = Scalar_;
        Matrix<Scalar, DynamicSize, DynamicSize> m_basis;

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        bool _is_linearly_independent(const Matrix<InScalar, InRows, InCols>& row) const {
            if (m_basis.rows() == 0) return true;
            Matrix<InScalar, InRows, InCols> ext = m_basis;
            ext.insert(row);
            return ext.rank() > m_basis.rows();
        }

    public:
        constexpr Span() noexcept : m_basis(0, 0) {}

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        explicit Span(const Matrix<InScalar, InRows, InCols>& mat)
            : m_basis(0, mat.cols()) {
            for (size_t i = 0; i < mat.rows(); ++i) {
                Matrix<Scalar, DynamicSize, DynamicSize> row = mat.submatrix(i, 0, 1, mat.cols());
                if (_is_linearly_independent(row)) insert(row);
            }
        }

        template<typename... InScalar, ptrdiff_t Size>
            requires(std::conjunction_v<std::is_convertible<InScalar, Vector<Scalar>>...>)
        Span(const Vector<InScalar, Size>&... vs)
            : m_basis(0, (sizeof...(InScalar) ? std::get<0>(std::tuple{ vs... }).size() : 0)) {
            (insert(vs), ...);
        }

        void insert(const Vector<Scalar>& v) {
            if (v.rank() == 0) return;
            Matrix<Scalar, DynamicSize, DynamicSize> row(1, v.size());
            for (size_t j = 0; j < v.size(); ++j) row(0, j) = v(0, j);
            if (_is_linearly_independent(row)) m_basis.insert(row);
        }

        template<typename InScalar, ptrdiff_t Size>
        constexpr bool contains(const Vector<InScalar, Size>& v) const {
            if (v.size() != m_basis.cols()) return false;
            if (v.rank() == 0) return true;
            Matrix<Scalar, DynamicSize, DynamicSize> ext = m_basis;
            ext.insert(v);
            return ext.rank() == m_basis.rows();
        }

        template<typename InScalar, ptrdiff_t InRows, ptrdiff_t InCols>
            requires(std::convertible_to<InScalar, Scalar>)
        constexpr bool contains(const Matrix<InScalar, InRows, InCols>& m) const {
            if (m.cols() != m_basis.cols()) return false;
            if (m.rank() == 0) return true;
            Matrix<Scalar, DynamicSize, DynamicSize> ext = m_basis;
            ext.insert(m);
            int r = ext.rank();
            return ext.rank() == m_basis.rows();
        }

        constexpr size_t dimension() const noexcept {
            return m_basis.rows();
        }
        constexpr auto& basis() const noexcept { return m_basis; }

        Matrix<Scalar, DynamicSize, DynamicSize> orthonormal() const {
            Scalar eps = Scalar(std::numeric_limits<double>::epsilon());
            size_t r = m_basis.rows();
            size_t c = m_basis.cols();
            Matrix<Scalar, DynamicSize, DynamicSize> Q(r, c);
            size_t k = 0;
            for (size_t i = 0; i < r; ++i) {
                Vector<Scalar> w = m_basis.submatrix(i, 0, 1, c);
                for (size_t t = 0; t < k; ++t) {
                    auto q = Q.submatrix(t, 0, 1, c);
                    Scalar proj = q.dot(w);
                    for (size_t j = 0; j < c; ++j) w(0, j) -= proj * Q(t, j);
                }
                Scalar norm2 = w.dot(w);
                auto norm = std::sqrt(norm2);
                if (norm > eps) {
                    for (size_t j = 0; j < c; ++j) Q(k, j) = w(0, j) / norm;
                    ++k;
                }
            }
            return Q.submatrix(0, 0, k, c);
        }
        Vector<Scalar> random() const {
            size_t r = m_basis.rows();
            size_t c = m_basis.cols();
            Matrix<Scalar> out(1, c);
            if (!r || !c) return out;
            std::mt19937_64 rng(std::random_device{}());
            std::uniform_real_distribution<double> dist(-1, 1);
            for (size_t i = 0; i < std::min(r, c); ++i) {
                auto coef = int(dist(rng) * 100);
                out(0, i) += coef * m_basis(i, i);
            }
            return out;
        }
    };

} // namespace linear

#endif // LINEAR_SPAN_H

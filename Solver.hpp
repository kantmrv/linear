//
//  Solver.hpp
//  linear
//
//  Created by Alexandr on 01.10.2025.
//

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <cmath>
#include <limits>

#include "Matrix.hpp"


namespace linear {

    // Base solver class for linear systems Ax = b
    template<FloatingPoint Scalar>
    class Base_Solver {
    public:
        virtual ~Base_Solver() = default;

        // Pure virtual solve method
        virtual Matrix<Scalar, DynamicSize, DynamicSize> solve(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b
        ) const = 0;
    };

    // Iterative solver base class with convergence control
    template<FloatingPoint Scalar>
    class Iter_Solver : public Base_Solver<Scalar> {
    protected:
        size_t m_max_iterations;
        Scalar m_tolerance;

    public:
        explicit Iter_Solver(size_t max_iter = 1000, Scalar tolerance = Scalar(1e-10))
            : m_max_iterations(max_iter), m_tolerance(tolerance) {}

        virtual ~Iter_Solver() = default;

        // Solve with initial guess
        virtual Matrix<Scalar, DynamicSize, DynamicSize> solve(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b,
            const Matrix<Scalar, DynamicSize, DynamicSize>& x0
        ) const = 0;

        // Solve with zero initial guess (override from Base_Solver)
        Matrix<Scalar, DynamicSize, DynamicSize> solve(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b
        ) const override {
            Matrix<Scalar, DynamicSize, DynamicSize> x0(b.rows(), b.cols());
            return solve(A, b, x0);
        }

        // A priori error estimate: ||e_k|| ≤ (||M||^k / (1 - ||M||)) * ||e_0||
        // where M = iteration matrix
        constexpr Scalar prior_estimate(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            size_t iterations
        ) const {
            // For Jacobi: M = -D^{-1}(L + U)
            // For Gauss-Seidel: M = -(D + L)^{-1}U
            // Simplified: estimate spectral radius as max off-diagonal ratio
            assert(A.rows() == A.cols());
            const size_t n = A.rows();

            Scalar max_ratio = Scalar(0);
            for (size_t i = 0; i < n; ++i) {
                Scalar off_diag_sum = Scalar(0);
                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        off_diag_sum += std::abs(A(i, j));
                    }
                }
                Scalar ratio = off_diag_sum / std::abs(A(i, i));
                max_ratio = std::max(max_ratio, ratio);
            }

            // Convergence requires spectral radius < 1
            assert(max_ratio < Scalar(1));

            return std::pow(max_ratio, static_cast<Scalar>(iterations)) /
                   (Scalar(1) - max_ratio);
        }

        // A posteriori error estimate: ||e_k|| ≤ (||M|| / (1 - ||M||)) * ||r_k||
        // where r_k = x_k - x_{k-1} is the residual between iterations
        constexpr Scalar posterior_estimate(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& x_curr,
            const Matrix<Scalar, DynamicSize, DynamicSize>& x_prev
        ) const {
            assert(A.rows() == A.cols());
            const size_t n = A.rows();

            // Compute ||x_curr - x_prev||
            Scalar residual = Scalar(0);
            for (size_t i = 0; i < n; ++i) {
                Scalar diff = x_curr(i, 0) - x_prev(i, 0);
                residual += diff * diff;
            }
            residual = std::sqrt(residual);

            // Estimate spectral radius
            Scalar max_ratio = Scalar(0);
            for (size_t i = 0; i < n; ++i) {
                Scalar off_diag_sum = Scalar(0);
                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        off_diag_sum += std::abs(A(i, j));
                    }
                }
                Scalar ratio = off_diag_sum / std::abs(A(i, i));
                max_ratio = std::max(max_ratio, ratio);
            }

            assert(max_ratio < Scalar(1));

            return (max_ratio / (Scalar(1) - max_ratio)) * residual;
        }

        // Getters and setters
        constexpr size_t max_iterations() const noexcept { return m_max_iterations; }
        constexpr Scalar tolerance() const noexcept { return m_tolerance; }

        constexpr void set_max_iterations(size_t max_iter) noexcept {
            m_max_iterations = max_iter;
        }
        constexpr void set_tolerance(Scalar tolerance) noexcept {
            m_tolerance = tolerance;
        }
    };

    // LU Solver using LU decomposition with partial pivoting
    template<FloatingPoint Scalar>
    class LU_Solver : public Base_Solver<Scalar> {
    public:
        // Solve Ax = b using LU decomposition
        // PA = LU, so Ax = b becomes LUx = Pb
        // Solve Ly = Pb, then Ux = y
        Matrix<Scalar, DynamicSize, DynamicSize> solve(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b
        ) const override {
            assert(A.rows() == A.cols());
            assert(b.rows() == A.rows());

            // Perform LU decomposition with partial pivoting
            auto [P, L, U] = A.lu();

            // Compute tilde_b = P * b
            Matrix<Scalar, DynamicSize, DynamicSize> b_tilde(b.rows(), b.cols());
            for (size_t i = 0; i < b.rows(); ++i) {
                for (size_t j = 0; j < b.cols(); ++j) {
                    b_tilde(i, j) = Scalar(0);
                    for (size_t k = 0; k < b.rows(); ++k) {
                        b_tilde(i, j) += P(i, k) * b(k, j);
                    }
                }
            }

            // Forward substitution: Ly = b_tilde
            auto y = L.forward_substitution(b_tilde);

            // Backward substitution: Ux = y
            auto x = U.backward_substitution(y);

            return x;
        }

        // Static convenience method
        static Matrix<Scalar, DynamicSize, DynamicSize> solve_system(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b
        ) {
            LU_Solver<Scalar> solver;
            return solver.solve(A, b);
        }
    };

    // QR Solver using QR decomposition
    template<FloatingPoint Scalar>
    class QR_Solver : public Base_Solver<Scalar> {
    public:
        // Solve Ax = b using QR decomposition
        // A = QR, so QRx = b
        // Multiply both sides by Q^T: Rx = Q^T * b
        // Solve Rx = Q^T * b using backward substitution
        Matrix<Scalar, DynamicSize, DynamicSize> solve(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b
        ) const override {
            assert(A.rows() == A.cols());
            assert(b.rows() == A.rows());

            // Perform QR decomposition
            auto [Q, R] = A.qr();

            // Compute b_tilde = Q^T * b
            Matrix<Scalar, DynamicSize, DynamicSize> b_tilde(Q.cols(), b.cols());
            for (size_t i = 0; i < Q.cols(); ++i) {
                for (size_t j = 0; j < b.cols(); ++j) {
                    b_tilde(i, j) = Scalar(0);
                    for (size_t k = 0; k < Q.rows(); ++k) {
                        b_tilde(i, j) += Q(k, i) * b(k, j);  // Q^T means Q(k,i) not Q(i,k)
                    }
                }
            }

            // Backward substitution: Rx = b_tilde
            auto x = R.backward_substitution(b_tilde);

            return x;
        }

        // Static convenience method
        static Matrix<Scalar, DynamicSize, DynamicSize> solve_system(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b
        ) {
            QR_Solver<Scalar> solver;
            return solver.solve(A, b);
        }
    };

    // Jacobi iterative solver
    template<FloatingPoint Scalar>
    class Jacobian_Solver : public Iter_Solver<Scalar> {
    public:
        using Iter_Solver<Scalar>::Iter_Solver;

        // Solve Ax = b using Jacobi iteration
        // x^{(k+1)}_i = (b_i - sum_{j≠i} a_{ij} * x^{(k)}_j) / a_{ii}
        Matrix<Scalar, DynamicSize, DynamicSize> solve(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b,
            const Matrix<Scalar, DynamicSize, DynamicSize>& x0
        ) const override {
            assert(A.rows() == A.cols());
            assert(b.rows() == A.rows());
            assert(x0.rows() == A.rows());
            assert(b.cols() == 1);  // b should be a column vector

            const size_t n = A.rows();

            // Check diagonal dominance (optional, for debugging)
            for (size_t i = 0; i < n; ++i) {
                assert(std::abs(A(i, i)) > std::numeric_limits<Scalar>::epsilon());
            }

            Matrix<Scalar, DynamicSize, DynamicSize> x_old = x0;
            Matrix<Scalar, DynamicSize, DynamicSize> x_new(n, 1);

            for (size_t k = 0; k < this->m_max_iterations; ++k) {
                // Jacobi iteration
                for (size_t i = 0; i < n; ++i) {
                    Scalar sum = Scalar(0);
                    for (size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += A(i, j) * x_old(j, 0);
                        }
                    }
                    x_new(i, 0) = (b(i, 0) - sum) / A(i, i);
                }

                // Check convergence: ||x_new - x_old||
                Scalar diff_norm = Scalar(0);
                for (size_t i = 0; i < n; ++i) {
                    Scalar diff = x_new(i, 0) - x_old(i, 0);
                    diff_norm += diff * diff;
                }
                diff_norm = std::sqrt(diff_norm);

                if (diff_norm < this->m_tolerance) {
                    return x_new;
                }

                x_old = x_new;
            }

            // Maximum iterations reached
            return x_new;
        }

        // Static convenience method
        static Matrix<Scalar, DynamicSize, DynamicSize> solve_system(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b,
            const Matrix<Scalar, DynamicSize, DynamicSize>& x0 = Matrix<Scalar, DynamicSize, DynamicSize>(),
            size_t max_iter = 1000,
            Scalar tolerance = Scalar(1e-10)
        ) {
            Jacobian_Solver<Scalar> solver(max_iter, tolerance);

            // If x0 is empty, create zero initial guess
            if (x0.rows() == 0 || x0.cols() == 0) {
                Matrix<Scalar, DynamicSize, DynamicSize> x0_default(b.rows(), b.cols());
                return solver.solve(A, b, x0_default);
            }

            return solver.solve(A, b, x0);
        }
    };

    // Gauss-Seidel iterative solver
    template<FloatingPoint Scalar>
    class Gauss_Seidel_Solver : public Iter_Solver<Scalar> {
    public:
        using Iter_Solver<Scalar>::Iter_Solver;

        // Solve Ax = b using Gauss-Seidel iteration
        // x^{(k+1)}_i = (b_i - sum_{j<i} a_{ij} * x^{(k+1)}_j - sum_{j>i} a_{ij} * x^{(k)}_j) / a_{ii}
        Matrix<Scalar, DynamicSize, DynamicSize> solve(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b,
            const Matrix<Scalar, DynamicSize, DynamicSize>& x0
        ) const override {
            assert(A.rows() == A.cols());
            assert(b.rows() == A.rows());
            assert(x0.rows() == A.rows());
            assert(b.cols() == 1);  // b should be a column vector

            const size_t n = A.rows();

            // Check diagonal dominance (optional, for debugging)
            for (size_t i = 0; i < n; ++i) {
                assert(std::abs(A(i, i)) > std::numeric_limits<Scalar>::epsilon());
            }

            Matrix<Scalar, DynamicSize, DynamicSize> x = x0;
            Matrix<Scalar, DynamicSize, DynamicSize> x_old(n, 1);

            for (size_t k = 0; k < this->m_max_iterations; ++k) {
                x_old = x;

                // Gauss-Seidel iteration
                for (size_t i = 0; i < n; ++i) {
                    Scalar sum1 = Scalar(0);  // Sum of already updated values
                    for (size_t j = 0; j < i; ++j) {
                        sum1 += A(i, j) * x(j, 0);
                    }

                    Scalar sum2 = Scalar(0);  // Sum of old values
                    for (size_t j = i + 1; j < n; ++j) {
                        sum2 += A(i, j) * x_old(j, 0);
                    }

                    x(i, 0) = (b(i, 0) - sum1 - sum2) / A(i, i);
                }

                // Check convergence: ||x - x_old||
                Scalar diff_norm = Scalar(0);
                for (size_t i = 0; i < n; ++i) {
                    Scalar diff = x(i, 0) - x_old(i, 0);
                    diff_norm += diff * diff;
                }
                diff_norm = std::sqrt(diff_norm);

                if (diff_norm < this->m_tolerance) {
                    return x;
                }
            }

            // Maximum iterations reached
            return x;
        }

        // Static convenience method
        static Matrix<Scalar, DynamicSize, DynamicSize> solve_system(
            const Matrix<Scalar, DynamicSize, DynamicSize>& A,
            const Matrix<Scalar, DynamicSize, DynamicSize>& b,
            const Matrix<Scalar, DynamicSize, DynamicSize>& x0 = Matrix<Scalar, DynamicSize, DynamicSize>(),
            size_t max_iter = 1000,
            Scalar tolerance = Scalar(1e-10)
        ) {
            Gauss_Seidel_Solver<Scalar> solver(max_iter, tolerance);

            // If x0 is empty, create zero initial guess
            if (x0.rows() == 0 || x0.cols() == 0) {
                Matrix<Scalar, DynamicSize, DynamicSize> x0_default(b.rows(), b.cols());
                return solver.solve(A, b, x0_default);
            }

            return solver.solve(A, b, x0);
        }
    };

} // namespace linear

#endif // SOLVER_HPP

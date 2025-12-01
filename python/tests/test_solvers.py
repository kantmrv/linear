"""
Test suite for linear system solvers
"""

import numpy as np

from linear import GaussSeidelSolver, JacobianSolver, LUSolver, Matrix, QRSolver


def test_lu_solver():
    """Test LU solver on simple system"""
    # System: [2 1] [x1]   [5]
    #         [1 3] [x2] = [6]
    # Solution: x1 = 1.8, x2 = 1.4
    A = Matrix(np.array([[2, 1], [1, 3]], dtype=np.float64))
    b = Matrix(np.array([[5], [6]], dtype=np.float64))

    solver = LUSolver()
    x = solver.solve(A, b)

    assert abs(x[0, 0] - 1.8) < 1e-10
    assert abs(x[1, 0] - 1.4) < 1e-10


def test_lu_solver_3x3():
    """Test LU solver on 3x3 system"""
    A = Matrix(np.array([[3, 2, 1], [2, 3, 2], [1, 2, 3]], dtype=np.float64))
    b = Matrix(np.array([[6], [7], [6]], dtype=np.float64))

    solver = LUSolver()
    x = solver.solve(A, b)

    # Verify Ax = b
    result = A * x
    for i in range(3):
        assert abs(result[i, 0] - b[i, 0]) < 1e-10


def test_qr_solver():
    """Test QR solver (more numerically stable)"""
    A = Matrix(np.array([[3, 2], [1, 2]], dtype=np.float64))
    b = Matrix(np.array([[5], [3]], dtype=np.float64))

    solver = QRSolver()
    x = solver.solve(A, b)

    # Verify Ax = b
    result = A * x
    assert abs(result[0, 0] - b[0, 0]) < 1e-10
    assert abs(result[1, 0] - b[1, 0]) < 1e-10


def test_qr_solver_3x3():
    """Test QR solver on 3x3 system"""
    A = Matrix(np.array([[4, 1, 2], [1, 5, 1], [2, 1, 6]], dtype=np.float64))
    b = Matrix(np.array([[7], [7], [9]], dtype=np.float64))

    solver = QRSolver()
    x = solver.solve(A, b)

    # Verify Ax = b
    result = A * x
    for i in range(3):
        assert abs(result[i, 0] - b[i, 0]) < 1e-10


def test_jacobian_solver():
    """Test Jacobi iterative solver on diagonally dominant system"""
    # Diagonally dominant: |a_ii| > sum(|a_ij|) for j != i
    A = Matrix(np.array([[5, 1], [1, 4]], dtype=np.float64))
    b = Matrix(np.array([[6], [5]], dtype=np.float64))

    solver = JacobianSolver(max_iterations=1000, tolerance=1e-10)
    x = solver.solve(A, b)

    # Verify solution
    result = A * x
    assert abs(result[0, 0] - b[0, 0]) < 1e-9
    assert abs(result[1, 0] - b[1, 0]) < 1e-9


def test_jacobian_solver_3x3():
    """Test Jacobi on 3x3 diagonally dominant system"""
    # Diagonally dominant matrix
    A = Matrix(np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]], dtype=np.float64))
    b = Matrix(np.array([[12], [12], [12]], dtype=np.float64))

    solver = JacobianSolver(max_iterations=500, tolerance=1e-10)
    x = solver.solve(A, b)

    # Verify solution
    result = A * x
    for i in range(3):
        assert abs(result[i, 0] - b[i, 0]) < 1e-8


def test_gauss_seidel_solver():
    """Test Gauss-Seidel (faster convergence than Jacobi)"""
    A = Matrix(np.array([[4, 1], [1, 3]], dtype=np.float64))
    b = Matrix(np.array([[5], [4]], dtype=np.float64))

    solver = GaussSeidelSolver(max_iterations=500, tolerance=1e-10)
    x = solver.solve(A, b)

    result = A * x
    assert abs(result[0, 0] - b[0, 0]) < 1e-9
    assert abs(result[1, 0] - b[1, 0]) < 1e-9


def test_gauss_seidel_3x3():
    """Test Gauss-Seidel on 3x3 system"""
    # Diagonally dominant
    A = Matrix(np.array([[8, 1, 1], [1, 7, 1], [1, 1, 6]], dtype=np.float64))
    b = Matrix(np.array([[10], [9], [8]], dtype=np.float64))

    solver = GaussSeidelSolver(max_iterations=300, tolerance=1e-10)
    x = solver.solve(A, b)

    # Verify solution
    result = A * x
    for i in range(3):
        assert abs(result[i, 0] - b[i, 0]) < 1e-8


def test_iterative_solver_properties():
    """Test iterative solver max_iterations and tolerance properties"""
    solver = JacobianSolver(max_iterations=500, tolerance=1e-8)

    assert solver.max_iterations == 500
    assert abs(solver.tolerance - 1e-8) < 1e-15

    # Test setter
    solver.max_iterations = 1000
    solver.tolerance = 1e-10

    assert solver.max_iterations == 1000
    assert abs(solver.tolerance - 1e-10) < 1e-15


def test_forward_backward_substitution():
    """Test triangular system solvers"""
    # Lower triangular
    L = Matrix(np.array([[2, 0], [1, 3]], dtype=np.float64))
    b = Matrix(np.array([[4], [7]], dtype=np.float64))
    x = L.forward_substitution(b)

    # Verify Lx = b
    result = L * x
    assert abs(result[0, 0] - 4.0) < 1e-10
    assert abs(result[1, 0] - 7.0) < 1e-10

    # Upper triangular
    U = Matrix(np.array([[2, 1], [0, 3]], dtype=np.float64))
    b2 = Matrix(np.array([[5], [6]], dtype=np.float64))
    x2 = U.backward_substitution(b2)

    # Verify Ux = b
    result2 = U * x2
    assert abs(result2[0, 0] - 5.0) < 1e-10
    assert abs(result2[1, 0] - 6.0) < 1e-10


def test_solver_comparison():
    """Compare direct and iterative solvers on same system"""
    # Diagonally dominant system
    A = Matrix(np.array([[5, 1, 0], [1, 6, 1], [0, 1, 7]], dtype=np.float64))
    b = Matrix(np.array([[6], [8], [8]], dtype=np.float64))

    # Solve with all methods
    lu_solver = LUSolver()
    qr_solver = QRSolver()
    jacobi_solver = JacobianSolver(max_iterations=1000, tolerance=1e-10)
    gs_solver = GaussSeidelSolver(max_iterations=500, tolerance=1e-10)

    x_lu = lu_solver.solve(A, b)
    x_qr = qr_solver.solve(A, b)
    x_jacobi = jacobi_solver.solve(A, b)
    x_gs = gs_solver.solve(A, b)

    # All solutions should be close
    for i in range(3):
        assert abs(x_lu[i, 0] - x_qr[i, 0]) < 1e-10
        assert abs(x_lu[i, 0] - x_jacobi[i, 0]) < 1e-8
        assert abs(x_lu[i, 0] - x_gs[i, 0]) < 1e-8

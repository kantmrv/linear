#include <iostream>

#include "Matrix.hpp"
#include "Span.hpp"


int main() {
    using namespace linear;

    // --- Matrix demonstrations ---
    auto A = Matrix2d( {1.0, 2.0}, {3.0, 4.0} );
    auto B = Matrix2d( {5.0, 6.0}, {7.0, 8.0} );

    std::cout << "A =" << '\n' << A << "\n\n";
    std::cout << "A + B =" << '\n' << (A + B) << "\n\n";
    std::cout << "A * B =" << '\n' << (A * B) << "\n\n";

    Vector2d v1 = { 1.0, 2.0 };
    Vector2d v2 = { 3.0, 4.0 };
    std::cout << "v1 * v2 = " << v1.dot(v2) << "\n\n";

    auto I = Matrix2d::Identity();
    std::cout << "Identity 2x2 =" << '\n' << I << "\n\n";

    // --- Span demonstrations ---
    Matrix<double, DynamicSize, DynamicSize> M = { {1.0, 0.0, 0.0},
                                                   {0.0, 1.0, 0.0},
                                                   {1.0, 1.0, 1.0} };
    Span span(M);
    std::cout << "Span dimension = " << span.dimension() << "\n\n";

    auto Q = span.orthonormal();
    std::cout << "Orthornormal basis =" << '\n' << Q << "\n\n";

    auto rvec = span.random();
    std::cout << "Random vector in span =" << '\n' << rvec << "\n\n";

    std::cout << "Span contains random vector " << int(span.contains(rvec)) << "\n\n";

    return 0;
}

#include <iostream>
#include <xtensor-sparse/xlil.hpp>

int main()
{
    xt::xlil_container<double> A;

    A(1, 0) = 3.;
    A(1, 2) = 10.;

    std::cout << A(1, 0) << "\n";
    std::cout << A(1, 2) << "\n";
    std::cout << A(3, 2) << "\n";

    return 0;
}
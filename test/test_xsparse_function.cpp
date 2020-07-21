#include "gtest/gtest.h"
#include <xtensor/xarray.hpp>

#include "xtensor-sparse/xsparse_function.hpp"
#include "xtensor-sparse/xsparse_array.hpp"

namespace xt
{
    TEST(xsparse_function, assign_tag)
    {
        xcoo_array<double> A;
        xarray<double> B;

        auto expr1 = A*B;
        EXPECT_TRUE((std::is_same<decltype(expr1)::assign_tag, extension::xsparse_assign_tag>::value));

        auto expr2 = A+B;
        EXPECT_TRUE((std::is_same<decltype(expr2)::assign_tag, extension::xdense_assign_tag>::value));

        auto expr3 = (A+B)*A;
        EXPECT_TRUE((std::is_same<decltype(expr3)::assign_tag, extension::xsparse_assign_tag>::value));

        auto expr4 = A+B*A;
        EXPECT_TRUE((std::is_same<decltype(expr4)::assign_tag, extension::xsparse_assign_tag>::value));

        auto expr5 = B+B*A;
        EXPECT_TRUE((std::is_same<decltype(expr5)::assign_tag, extension::xdense_assign_tag>::value));

        auto expr6 = (A+B)*B;
        EXPECT_TRUE((std::is_same<decltype(expr6)::assign_tag, extension::xdense_assign_tag>::value));
    }

    TEST(xsparse_function, iterator)
    {
        std::vector<std::size_t> shape{2, 5};
        xcoo_array<double> A(shape);
        xcoo_array<double> C(shape);
        xarray<double> B;

        A(0, 2) = 2.5;
        A(1, 2) = 1.1;
        C(1, 2) = 3.2;
        auto expr1 = A*C;
        auto it = expr1.nz_begin();
        std::cout << *it << "\n";
        ++it;
        std::cout << *it << "\n";
        // auto it = xfunction_nz_iterator(expr1)
    }
}
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
}
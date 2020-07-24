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

    TEST(xsparse_function, nz_iterator_begin)
    {
        std::vector<std::size_t> shape{2, 5};
        xcoo_array<double> A(shape);
        xcsf_array<double> B(shape);

        A(0, 2) = 2.1;
        A(1, 2) = 1.2;
        B(1, 2) = 4.2;

        auto expr1 = A*B + B;

        auto it1 = expr1.nz_begin();
        EXPECT_EQ(*it1, 0.);
        ++it1;
        EXPECT_EQ(*it1, 9.24);

        auto expr2 = A*B + A;
        auto it2 = expr2.nz_begin();
        EXPECT_EQ(*it2, 2.1);
        ++it2;
        EXPECT_EQ(*it2, 6.24);

    }

    TEST(xsparse_function, nz_iterator_end)
    {
        std::vector<std::size_t> shape{2, 5};
        xcoo_array<double> A(shape);
        xcsf_array<double> B(shape);

        A(0, 2) = 2.1;
        A(1, 2) = 1.2;
        B(1, 2) = 4.2;

        auto expr1 = A*B + B;

        auto it1 = expr1.nz_end();
        --it1;
        EXPECT_EQ(*it1, 9.24);
        --it1;
        EXPECT_EQ(*it1, 0.);

        auto expr2 = A*B + A;
        auto it2 = expr2.nz_end();
        --it2;
        EXPECT_EQ(*it2, 6.24);
        --it2;
        EXPECT_EQ(*it2, 2.1);

    }

}
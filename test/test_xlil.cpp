#include "gtest/gtest.h"
#include <iostream>
#include <vector>
#include <xtensor-sparse/xlil.hpp>

namespace xt
{
    TEST(xlil, shaped_constructor)
    {
        std::vector<std::size_t> shape{2, 5};
        xt::xlil_container<double> A(shape);

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(2));
        EXPECT_EQ(A.shape()[1], size_t(5));
    }

    TEST(xlil, access_operator)
    {
        std::vector<std::size_t> shape{2, 5};
        xt::xlil_container<double> A(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        EXPECT_EQ(A(0, 0), 3.);
        EXPECT_EQ(A(1, 2), 10.);
        EXPECT_EQ(A(1, 4), 0.);
    }
}


#include "gtest/gtest.h"
#include <vector>
#include <xtensor-sparse/xcoo_tensor.hpp>

namespace xt
{
    /*TEST(xcoo_tensor, shaped_constructor)
    {
        xt::xcoo_tensor<double, 2> A({2, 5});

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(2));
        EXPECT_EQ(A.shape()[1], size_t(5));
    }

    TEST(xcoo_tensor, resize)
    {
        xt::xcoo_tensor<double, 2> A({2, 5});

        std::vector<std::size_t> new_shape{20, 50};
        A.resize(new_shape);
        EXPECT_EQ(A.shape()[0], size_t(20));
        EXPECT_EQ(A.shape()[1], size_t(50));
    }

    TEST(xcoo_tensor, reshape_unsigned)
    {
        xt::xcoo_tensor<double, 2> A({10, 1});

        A(1) = 1.;
        A(5) = 5.;
        A(7) = 7.;

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(10));

        std::vector<std::size_t> new_shape{2, 5};
        A.reshape(new_shape);

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(2));
        EXPECT_EQ(A.shape()[1], size_t(5));

        EXPECT_EQ(A(0, 1), 1.);
        EXPECT_EQ(A(1, 0), 5.);
        EXPECT_EQ(A(1, 2), 7.);
    }

    TEST(xcoo_tensor, reshape_signed)
    {
        xt::xcoo_tensor<double, 2> A({20, 1});

        A(1) = 1.;
        A(5) = 5.;
        A(7) = 7.;

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(20));

        std::vector<int> new_shape{-1, 5};
        A.reshape(new_shape);

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(4));
        EXPECT_EQ(A.shape()[1], size_t(5));

        EXPECT_EQ(A(0, 1), 1.);
        EXPECT_EQ(A(1, 0), 5.);
        EXPECT_EQ(A(1, 2), 7.);
    }

    TEST(xcoo_tensor, access_operator)
    {
        xt::xcoo_tensor<double, 2> A({2, 5});

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        EXPECT_EQ(A(0, 0), 3.);
        EXPECT_EQ(A(1, 2), 10.);
        EXPECT_EQ(A(1, 4), 0.);
    }*/
}


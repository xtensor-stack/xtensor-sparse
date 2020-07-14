#include "gtest/gtest.h"
#include <vector>
#include <xtensor-sparse/xsparse_array.hpp>

namespace xt
{
    TEST(xmap_array, shaped_constructor)
    {
        std::vector<std::size_t> shape{2, 5};
        xt::xmap_array<double> A(shape);

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(2));
        EXPECT_EQ(A.shape()[1], size_t(5));
    }

    TEST(xmap_array, resize)
    {
        std::vector<std::size_t> shape{2, 5};
        xt::xmap_array<double> A(shape);

        std::vector<std::size_t> new_shape{20, 50};
        A.resize(new_shape);
        EXPECT_EQ(A.shape()[0], size_t(20));
        EXPECT_EQ(A.shape()[1], size_t(50));
    }

    TEST(xmap_array, reshape_unsigned)
    {
        std::vector<std::size_t> shape{10};
        xt::xmap_array<double> A(shape);

        A(1) = 1.;
        A(5) = 5.;
        A(7) = 7.;

        EXPECT_EQ(A.dimension(), size_t(1));
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

    TEST(xmap_array, reshape_signed)
    {
        std::vector<std::size_t> shape{20};
        xt::xmap_array<double> A(shape);

        A(1) = 1.;
        A(5) = 5.;
        A(7) = 7.;

        EXPECT_EQ(A.dimension(), size_t(1));
        EXPECT_EQ(A.shape()[0], size_t(20));

        std::vector<int> new_shape{2, -1, 5};
        A.reshape(new_shape);

        EXPECT_EQ(A.dimension(), size_t(3));
        EXPECT_EQ(A.shape()[0], size_t(2));
        EXPECT_EQ(A.shape()[1], size_t(2));
        EXPECT_EQ(A.shape()[2], size_t(5));

        EXPECT_EQ(A(0, 0, 1), 1.);
        EXPECT_EQ(A(0, 1, 0), 5.);
        EXPECT_EQ(A(0, 1, 2), 7.);
    }

    TEST(xmap_array, access_operator)
    {
        std::vector<std::size_t> shape{2, 5};
        xt::xmap_array<double> A(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        EXPECT_EQ(A(0, 0), 3.);
        EXPECT_EQ(A(1, 2), 10.);
        EXPECT_EQ(A(1, 4), 0.);
    }
}


#include "gtest/gtest.h"
#include <vector>
#include <xtensor-sparse/xcoo_array.hpp>
#include <xtensor-sparse/xcoo_tensor.hpp>
#include <xtensor-sparse/xcsf_array.hpp>

namespace xt
{
    template <class S>
    class container_test : public ::testing::Test
    {
    public:

        using scheme_type = S;
    };

    using container_list_types = ::testing::Types<xcoo_array<double>,
                                                //   xcoo_tensor<double, 2>,
                                                  xcsf_array<double>>;
    TYPED_TEST_SUITE(container_test, container_list_types);

    TYPED_TEST(container_test, shaped_constructor)
    {
        std::vector<std::size_t> shape{2, 5};
        TypeParam A(shape);

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(2));
        EXPECT_EQ(A.shape()[1], size_t(5));
    }

    TYPED_TEST(container_test, at)
    {
        std::vector<std::size_t> shape{2, 5};
        TypeParam A(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        EXPECT_EQ(A.at(0, 0), 3.);
        EXPECT_EQ(A.at(0, 1), 0.);
        EXPECT_EQ(A.at(1, 2), 10.);
    }

    TYPED_TEST(container_test, shape)
    {
        std::vector<std::size_t> shape{2, 5};
        TypeParam A(shape);

        EXPECT_EQ(A.shape(0), 2);
        EXPECT_EQ(A.shape(1), 5);
    }

    TYPED_TEST(container_test, element)
    {
        std::vector<std::size_t> shape{2, 5};
        TypeParam A(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        std::array<std::size_t, 2> index{0, 0};
        EXPECT_EQ(A.element(index.cbegin(), index.cend()), 3);
        index = {1, 2};
        EXPECT_EQ(A.element(index.cbegin(), index.cend()), 10);
        index = {0, 2};
        EXPECT_EQ(A.element(index.cbegin(), index.cend()), 0);
    }
}
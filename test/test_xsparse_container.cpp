#include "gtest/gtest.h"

#include <tuple>
#include "test_common.hpp"


namespace xt
{
    template <class S>
    class container_test : public ::testing::Test
    {};

    TYPED_TEST_SUITE(container_test, container_list_types);

    TYPED_TEST(container_test, shaped_constructor)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using shape_type = typename xsparse_type::shape_type;

        shape_type shape{2, 5};
        xsparse_type A(shape);

        EXPECT_EQ(A.dimension(), size_t(2));
        EXPECT_EQ(A.shape()[0], size_t(2));
        EXPECT_EQ(A.shape()[1], size_t(5));
    }

    TYPED_TEST(container_test, at)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using shape_type = typename xsparse_type::shape_type;

        shape_type shape{2, 5};
        xsparse_type A(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        EXPECT_EQ(A.at(0, 0), 3.);
        EXPECT_EQ(A.at(0, 1), 0.);
        EXPECT_EQ(A.at(1, 2), 10.);
    }

    TYPED_TEST(container_test, shape)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using shape_type = typename xsparse_type::shape_type;

        shape_type shape{2, 5};
        xsparse_type A(shape);

        EXPECT_EQ(A.shape(0), 2);
        EXPECT_EQ(A.shape(1), 5);
    }

    TYPED_TEST(container_test, element)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using shape_type = typename xsparse_type::shape_type;

        shape_type shape{2, 5};
        xsparse_type A(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        std::array<std::size_t, 2> index{0, 0};
        EXPECT_EQ(A.element(index.cbegin(), index.cend()), 3);
        index = {1, 2};
        EXPECT_EQ(A.element(index.cbegin(), index.cend()), 10);
        index = {0, 2};
        EXPECT_EQ(A.element(index.cbegin(), index.cend()), 0);
    }

    TYPED_TEST(container_test, iterator)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using shape_type = typename xsparse_type::shape_type;

        shape_type shape{2, 5};
        xsparse_type A(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;

        auto it = A.begin();
        EXPECT_EQ(*it, 3);
        ++it;
        EXPECT_EQ(*it, 0.);
        *it = 7.8;
        EXPECT_EQ(*it, 7.8);
        EXPECT_EQ(A(0, 1), 7.8);
        it += 6;
        EXPECT_EQ(*it, 10.);
        *it = 5.6;
        EXPECT_EQ(*it, 5.6);
        it += 3;
        EXPECT_EQ(it, A.end());
    }

    TYPED_TEST(container_test, semantic)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using shape_type = typename xsparse_type::shape_type;

        shape_type shape{2, 5};
        xsparse_type A(shape), B(shape);

        A(0, 0) = 3.;
        A(1, 2) = 10.;
        A(1, 0) = 20.;

        B(0, 0) = 7.;
        B(1, 4) = 9.;

        EXPECT_EQ(A.dimension(), size_t(2));

        B = 2*A + B;
        EXPECT_EQ(B(0, 0), 13.);
        EXPECT_EQ(B(1, 2), 20.);
        EXPECT_EQ(B(1, 0), 40.);

        EXPECT_EQ(B(0, 2), 0.);
        EXPECT_EQ(B(1, 4), 9.);

        B = 2*A + 1;
        EXPECT_EQ(B(0, 0), 7.);
        EXPECT_EQ(B(1, 2), 21.);
        EXPECT_EQ(B(1, 0), 41.);

        EXPECT_EQ(B(0, 2), 1.);
        EXPECT_EQ(B(1, 4), 1.);
    }
}

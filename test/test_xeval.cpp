#include "gtest/gtest.h"

#include <tuple>
#include "test_common.hpp"

#include <xtensor/xeval.hpp>
#include <xtensor-sparse/xsparse_tensor_traits.hpp>

namespace xt
{
    template <class S>
    class xeval_test : public ::testing::Test
    {};

    TYPED_TEST_SUITE(xeval_test, container_list_types);

    TYPED_TEST(xeval_test, sparse_result)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using xtensor_type = typename std::tuple_element<1, TypeParam>::type;
        using xdefault_sparse_type = typename std::tuple_element<2, TypeParam>::type;

        using shape_type = typename xsparse_type::shape_type;
        shape_type shape{2, 5};
        xsparse_type A(shape);

        A(1, 3) = 2.;
        auto result = eval(2*A);

        bool type_eq = std::is_same<decltype(result), xdefault_sparse_type>::value;
        EXPECT_TRUE(type_eq);

        EXPECT_EQ(result(1, 3), 4.);
    }

    TYPED_TEST(xeval_test, dense_result)
    {
        using xsparse_type = typename std::tuple_element<0, TypeParam>::type;
        using xtensor_type = typename std::tuple_element<1, TypeParam>::type;
        using shape_type = typename xsparse_type::shape_type;
        shape_type shape{2, 5};
        xsparse_type A(shape);

        A(1, 3) = 2.;
        auto result = eval(2*A + 1);

        bool type_eq = std::is_same<decltype(result), xtensor_type>::value;
        EXPECT_TRUE(type_eq);

        EXPECT_EQ(result(1, 3), 5.);
        EXPECT_EQ(result(0, 0), 1.);
        EXPECT_EQ(result(1, 2), 1.);

    }
}
#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

#include <tuple>
#include <xtensor-sparse/xsparse_array.hpp>
#include <xtensor-sparse/xsparse_tensor.hpp>

namespace xt
{
    using container_list_types = ::testing::Types<
                                 std::tuple<    xcoo_array<double>,     xarray<double>,     xcoo_array<double>>,
                                 std::tuple<xcoo_tensor<double, 2>, xtensor<double, 2>, xcoo_tensor<double, 2>>,
                                 std::tuple<    xcsf_array<double>,     xarray<double>,     xcoo_array<double>>,
                                 std::tuple<xcsf_tensor<double, 2>, xtensor<double, 2>, xcoo_tensor<double, 2>>,
                                 std::tuple<    xmap_array<double>,     xarray<double>,     xcoo_array<double>>,
                                 std::tuple<xmap_tensor<double, 2>, xtensor<double, 2>, xcoo_tensor<double, 2>>>;
}

#endif
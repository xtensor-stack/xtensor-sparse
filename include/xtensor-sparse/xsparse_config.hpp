#ifndef XTENSOR_SPARSE_CONFIG_HPP
#define XTENSOR_SPARSE_CONFIG_HPP

#define XTENSOR_SPARSE_VERSION_MAJOR 0
#define XTENSOR_SPARSE_VERSION_MINOR 0
#define XTENSOR_SPARSE_VERSION_PATCH 1

#define XSPARSE_DEFAULT_ARRAY_SCHEME(SCHEME, T) \
    xt::xdefault_##SCHEME##_scheme_t<T, svector<std::size_t>>

#define XSPARSE_DEFAULT_TENSOR_SCHEME(SCHEME, T, N) \
    xt::xdefault_##SCHEME##_scheme_t<T, std::array<std::size_t, N>>

#define XSPARSE_DEFAULT_ARRAY(T) \
    xsparse_array<T, XSPARSE_DEFAULT_ARRAY_SCHEME(coo, T)>

#define XSPARSE_DEFAULT_TENSOR(T, N) \
    xsparse_tensor<T, N, XSPARSE_DEFAULT_TENSOR_SCHEME(coo, T, N)>

#endif

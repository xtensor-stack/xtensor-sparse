#ifndef XSPARSE_TYPES_HPP
#define XSPARSE_TYPES_HPP

#include "xcoo_scheme.hpp"
#include "xcsf_scheme.hpp"
#include "xmap_scheme.hpp"
#include "xsparse_config.hpp"

namespace xt
{

    template <class T, class S>
    class xsparse_array;

    template <class T, std::size_t N, class S>
    class xsparse_tensor;

    /*****************************
     * Common sparse array types *
     *****************************/

    template <class T>
    using xcoo_array = xsparse_array<T, XSPARSE_DEFAULT_ARRAY_SCHEME(coo, T)>;

    template <class T>
    using xcsf_array = xsparse_array<T, XSPARSE_DEFAULT_ARRAY_SCHEME(csf, T)>;

    template <class T>
    using xmap_array = xsparse_array<T, XSPARSE_DEFAULT_ARRAY_SCHEME(map, T)>;

    /******************************
     * Common sparse tensor types *
     ******************************/

    template <class T, std::size_t N>
    using xcoo_tensor = xsparse_tensor<T, N, XSPARSE_DEFAULT_TENSOR_SCHEME(coo, T, N)>;

    template <class T, std::size_t N>
    using xcsf_tensor = xsparse_tensor<T, N, XSPARSE_DEFAULT_TENSOR_SCHEME(csf, T, N)>;

    template <class T, std::size_t N>
    using xmap_tensor = xsparse_tensor<T, N, XSPARSE_DEFAULT_TENSOR_SCHEME(map, T, N)>;
}
#endif
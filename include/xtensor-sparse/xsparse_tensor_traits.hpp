#ifndef XSPARSE_XSPARSE_TENSOR_TRAITS_HPP
#define XSPARSE_XSPARSE_TENSOR_TRAITS_HPP

// #include <xtensor-sparse/xsparse_tensor_traits.hpp>

#include "xsparse_expression.hpp"
#include "xsparse_array.hpp"
#include "xsparse_tensor.hpp"

namespace xt
{
    namespace detail
    {
        template <class S>
        struct xsparse_type_for_shape
        {
            template <class T>
            using type = xcoo_array<T>;
        };

#if defined(__GNUC__) && (__GNUC__ > 6)
#if __cplusplus == 201703L
        template <template <class, std::size_t, class, bool> class S, class X, std::size_t N, class A, bool Init>
        struct xsparse_type_for_shape<S<X, N, A, Init>>
        {
            template <class T>
            using type = xcoo_array<T>;
        };
#endif // __cplusplus == 201703L
#endif // __GNUC__ && (__GNUC__ > 6)

        template <template <class, std::size_t> class S, class X, std::size_t N>
        struct xsparse_type_for_shape<S<X, N>>
        {
            template <class T>
            using type = xcoo_tensor<T, N>;
        };
    }

    template <class Tag, class T>
    struct temporary_type_from_tag;

    template <class T>
    struct temporary_type_from_tag<xsparse_expression_tag, T>
    {
        using I = std::decay_t<T>;

        using shape_type = typename I::shape_type;
        using value_type = typename I::value_type;

        using type = std::conditional_t<std::is_same<typename   I::assign_tag, extension::xdense_assign_tag>::value,
                                        typename temporary_type_from_tag<xtensor_expression_tag, T>::type,
                                        typename detail::xsparse_type_for_shape<shape_type>::template type<value_type>>;
    };
}

#endif
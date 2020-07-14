#ifndef XSPARSE_XSPARSE_TENSOR_HPP
#define XSPARSE_XSPARSE_TENSOR_HPP

#include <xtensor/xiterator.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsemantic.hpp>
#include <xtensor/xstorage.hpp>

#include "xcoo_scheme.hpp"
#include "xcsf_scheme.hpp"
#include "xcsr_scheme.hpp"
#include "xmap_scheme.hpp"
#include "xsparse_container.hpp"

namespace xt
{
    /******************
     * xsparse_tensor *
     ******************/

    template <class T, std::size_t N, class S>
    class xsparse_tensor;

    template <class T, std::size_t N, class S>
    struct xcontainer_inner_types<xsparse_tensor<T, N, S>>
    {
        using scheme_type = S;
        using storage_type = typename scheme_type::storage_type;
        using index_type = typename scheme_type::index_type;
        using value_type = T;
        using reference = xsparse_reference<scheme_type>;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using shape_type = std::array<size_type, N>;
        using strides_type = std::array<size_type, N>;
        using inner_shape_type = shape_type;

        using temporary_type = xsparse_tensor<T, N, S>;
    };

    template <class T, std::size_t N, class S>
    struct xiterable_inner_types<xsparse_tensor<T, N, S>>
    {
        using tensor_type = xsparse_tensor<T, N, S>;
        using inner_shape_type = typename xcontainer_inner_types<tensor_type>::shape_type;
        using const_stepper = xindexed_stepper<tensor_type, true>;
        using stepper = xindexed_stepper<tensor_type, false>;
    };

    template <class T, std::size_t N, class S>
    class xsparse_tensor : public xsparse_container<xsparse_tensor<T, N, S>>,
                           public xcontainer_semantic<xsparse_tensor<T, N, S>>
    {
    public:

        using self_type = xsparse_tensor<T, N, S>;
        using base_type = xsparse_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using temporary_type = typename base_type::temporary_type;

        xsparse_tensor() = default;
        explicit xsparse_tensor(const shape_type& shape);

        ~xsparse_tensor() = default;

        xsparse_tensor(const xsparse_tensor&) = default;
        xsparse_tensor& operator=(const xsparse_tensor&) = default;

        xsparse_tensor(xsparse_tensor&&) = default;
        xsparse_tensor& operator=(xsparse_tensor&&) = default;

        template <class E>
        xsparse_tensor(const xexpression<E>& e);

        template <class E>
        xsparse_tensor& operator=(const xexpression<E>& e);
 
    };

    /******************************
     * Common sparse tensor types *
     ******************************/

    template <class T, std::size_t N>
    using xcoo_tensor = xsparse_tensor<T, N, xdefault_coo_scheme_t<T, std::array<std::size_t, N>>>;

    template <class T, std::size_t N>
    using xcsf_tensor = xsparse_tensor<T, N, xdefault_csf_scheme_t<T, std::array<std::size_t, N>>>;

    template <class T, std::size_t N>
    using xmap_tensor = xsparse_tensor<T, N, xdefault_map_scheme_t<T, std::array<std::size_t, N>>>;

    /*********************************
     * xsparse_tensor implementation *
     *********************************/

    template <class T, std::size_t N, class S>
    inline xsparse_tensor<T, N, S>::xsparse_tensor(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }
    template <class T, std::size_t N, class S>
    template <class E>
    inline xsparse_tensor<T, N, S>::xsparse_tensor(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    template <class T, std::size_t N, class S>
    template <class E>
    inline auto xsparse_tensor<T, N, S>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
}

#endif

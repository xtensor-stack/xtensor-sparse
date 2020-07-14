#ifndef XSPARSE_XSPARSE_ARRAY_HPP
#define XSPARSE_XSPARSE_ARRAY_HPP

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
    /*****************
     * xsparse_array *
     *****************/

    template <class T, class S>
    class xsparse_array;

    template <class T, class S>
    struct xcontainer_inner_types<xsparse_array<T, S>>
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

        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;
        using inner_shape_type = shape_type;

        using temporary_type = xsparse_array<T, S>;
    };

    template <class T, class S>
    struct xiterable_inner_types<xsparse_array<T, S>>
    {
        using array_type = xsparse_array<T, S>;
        using inner_shape_type = typename xcontainer_inner_types<array_type>::shape_type;
        using const_stepper = xindexed_stepper<array_type, true>;
        using stepper = xindexed_stepper<array_type, false>;
    };

    template <class T, class S>
    class xsparse_array : public xsparse_container<xsparse_array<T, S>>,
                          public xcontainer_semantic<xsparse_array<T, S>>
    {
    public:

        using self_type = xsparse_array<T, S>;
        using base_type = xsparse_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using temporary_type = typename base_type::temporary_type;

        xsparse_array() = default;
        explicit xsparse_array(const shape_type& shape);

        ~xsparse_array() = default;

        xsparse_array(const xsparse_array&) = default;
        xsparse_array& operator=(const xsparse_array&) = default;

        xsparse_array(xsparse_array&&) = default;
        xsparse_array& operator=(xsparse_array&&) = default;

        template <class E>
        xsparse_array(const xexpression<E>& e);

        template <class E>
        xsparse_array& operator=(const xexpression<E>& e);
    };

    /*****************************
     * Common sparse array types *
     *****************************/

    template <class T>
    using xcoo_array = xsparse_array<T, xdefault_coo_scheme_t<T, svector<std::size_t>>>;
    
    template <class T>
    using xcsf_array = xsparse_array<T, xdefault_csf_scheme_t<T, svector<std::size_t>>>;

    template <class T>
    using xmap_array = xsparse_array<T, xdefault_map_scheme_t<T, svector<std::size_t>>>;

    /********************************
     * xsparse_array implementation *
     ********************************/

    template <class T, class S>
    inline xsparse_array<T, S>::xsparse_array(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }

    template <class T, class S>
    template <class E>
    inline xsparse_array<T, S>::xsparse_array(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    template <class T, class S>
    template <class E>
    inline auto xsparse_array<T, S>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
}

#endif


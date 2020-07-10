#ifndef XSPARSE_MAP_ARRAY_HPP
#define XSPARSE_MAP_ARRAY_HPP

#include <map>
#include <xtensor/xiterator.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsemantic.hpp>
#include <xtensor/xstorage.hpp>

#include "xmap_scheme.hpp"
#include "xsparse_container.hpp"

namespace xt
{
    
    /**************************
     * xmap_array declaration *
     **************************/

    template <class T>
    class xmap_array;

    template<class T>
    struct xcontainer_inner_types<xmap_array<T>>
    {
        using value_type = T;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;
        using inner_shape_type = shape_type;

        using index_type = svector<size_type>;
        using storage_type = std::map<index_type, value_type>;
        using scheme_type = xmap_scheme<storage_type>;

        using reference = xsparse_reference<scheme_type>;
        using temporary_type = xmap_array<T>;
    };
    
    template <class T>
    struct xiterable_inner_types<xmap_array<T>>
    {
        using array_type = xmap_array<T>;
        using inner_shape_type = typename xcontainer_inner_types<array_type>::shape_type;
        using const_stepper = xindexed_stepper<array_type, true>;
        using stepper = xindexed_stepper<array_type, false>;
    };

    template <class T>
    class xmap_array : public xsparse_container<xmap_array<T>>,
                       public xcontainer_semantic<xmap_array<T>>
    {
    public:

        using self_type = xmap_array<T>;
        using base_type = xsparse_container<self_type>;
        using storage_type = typename base_type::storage_type;
        using index_type = typename base_type::index_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using temporary_type = typename base_type::temporary_type;
        using inner_shape_type = typename base_type::inner_shape_type;

        xmap_array();
        explicit xmap_array(const shape_type& shape);

        ~xmap_array() = default;

        xmap_array(const xmap_array&) = default;
        xmap_array& operator=(const xmap_array&) = default;

        xmap_array(xmap_array&&) = default;
        xmap_array& operator=(xmap_array&&) = default;
    };

    /*****************************
     * xmap_array implementation *
     *****************************/

    template<class T>
    inline xmap_array<T>::xmap_array()
        : base_type()
    {
    }

    template<class T>
    inline xmap_array<T>::xmap_array(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }
}

#endif


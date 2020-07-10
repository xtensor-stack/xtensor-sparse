#ifndef XSPARSE_COO_ARRAY_HPP
#define XSPARSE_COO_ARRAY_HPP

#include <array>
#include <vector>
#include <xtensor/xiterator.hpp>
#include <xtensor/xstorage.hpp>

#include "xcoo_scheme.hpp"
#include "xsparse_container.hpp"

namespace xt
{
    /**************************
     * xcoo_array declaration *
     **************************/

    template <class T>
    class xcoo_array;

    template<class T>
    struct xcontainer_inner_types<xcoo_array<T>>
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
        using storage_type = std::vector<value_type>;
        using scheme_type = xcoo_scheme<std::array<size_type, 2>,
                                        std::vector<index_type>,
                                        storage_type,
                                        index_type>;
        using reference = xsparse_reference<scheme_type>;
    };
    
    template <class T>
    struct xiterable_inner_types<xcoo_array<T>>
    {
        using array_type = xcoo_array<T>;
        using inner_shape_type = typename xcontainer_inner_types<array_type>::shape_type;
        using const_stepper = xindexed_stepper<array_type, true>;
        using stepper = xindexed_stepper<array_type, false>;
    };

    template <class T>
    class xcoo_array : public xsparse_container<xcoo_array<T>> 
    {
    public:

        using self_type = xcoo_array<T>;
        using base_type = xsparse_container<self_type>;
        using storage_type = typename base_type::storage_type;
        using index_type = typename base_type::index_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;

        xcoo_array();
        explicit xcoo_array(const shape_type& shape);

        ~xcoo_array() = default;

        xcoo_array(const xcoo_array&) = default;
        xcoo_array& operator=(const xcoo_array&) = default;

        xcoo_array(xcoo_array&&) = default;
        xcoo_array& operator=(xcoo_array&&) = default;
    };

    /*****************************
     * xcoo_array implementation *
     *****************************/

    template<class T>
    inline xcoo_array<T>::xcoo_array()
        : base_type()
    {
    }

    template<class T>
    inline xcoo_array<T>::xcoo_array(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }
}

#endif

#ifndef XSPARSE_COO_TENSOR_HPP
#define XSPARSE_COO_TENSOR_HPP

#include <xtensor/xiterator.hpp>

#include "xcoo_scheme.hpp"
#include "xsparse_container.hpp"

namespace xt
{
    /***************************
     * xcoo_tensor declaration *
     ***************************/

    template <class T, std::size_t N>
    class xcoo_tensor;

    template<class T, std::size_t N>
    struct xcontainer_inner_types<xcoo_tensor<T, N>>
    {
        using value_type = T;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using shape_type = std::array<size_type, N>;
        using strides_type = std::array<size_type, N>;
        using inner_shape_type = shape_type;

        using index_type = std::array<size_type, N>;
        using storage_type = std::vector<value_type>;
        using scheme_type = xcoo_scheme<std::array<size_type, 2>,
                                        std::vector<index_type>,
                                        storage_type,
                                        index_type>;
        using reference = xsparse_reference<scheme_type>;
    };
    
    template <class T, std::size_t N>
    struct xiterable_inner_types<xcoo_tensor<T, N>>
    {
        using array_type = xcoo_tensor<T, N>;
        using inner_shape_type = typename xcontainer_inner_types<array_type>::shape_type;
        using const_stepper = xindexed_stepper<array_type, true>;
        using stepper = xindexed_stepper<array_type, false>;
    };

    template <class T, std::size_t N>
    class xcoo_tensor : public xsparse_container<xcoo_tensor<T, N>> 
    {
    public:

        using self_type = xcoo_tensor<T, N>;
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

        xcoo_tensor();
        explicit xcoo_tensor(const shape_type& shape);

        ~xcoo_tensor() = default;

        xcoo_tensor(const xcoo_tensor&) = default;
        xcoo_tensor& operator=(const xcoo_tensor&) = default;

        xcoo_tensor(xcoo_tensor&&) = default;
        xcoo_tensor& operator=(xcoo_tensor&&) = default;
    };

    /******************************
     * xcoo_tensor implementation *
     ******************************/

    template<class T, std::size_t N>
    inline xcoo_tensor<T, N>::xcoo_tensor()
        : base_type()
    {
    }

    template<class T, std::size_t N>
    inline xcoo_tensor<T, N>::xcoo_tensor(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }
}

#endif

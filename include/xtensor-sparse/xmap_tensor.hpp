#ifndef XSPARSE_MAP_TENSOR_HPP
#define XSPARSE_MAP_TENSOR_HPP

#include <array>
#include <map>

#include <xtensor/xiterator.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsemantic.hpp>

#include "xmap_scheme.hpp"
#include "xsparse_container.hpp"

namespace xt
{

    /***************************
     * xmap_tensor declaration *
     ***************************/

    template<class T, std::size_t N>
    class xmap_tensor;

    template<class T, std::size_t N>
    struct xcontainer_inner_types<xmap_tensor<T, N>>
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
        using storage_type = std::map<index_type, value_type>;
        using scheme_type = xmap_scheme<storage_type>;

        using reference = xsparse_reference<scheme_type>;
        using temporary_type = xmap_tensor<T, N>;
    };

    template <class T, std::size_t N>
    struct xiterable_inner_types<xmap_tensor<T, N>>
    {
        using array_type = xmap_tensor<T, N>;
        using inner_shape_type = typename xcontainer_inner_types<array_type>::shape_type;
        using const_stepper = xindexed_stepper<array_type, true>;
        using stepper = xindexed_stepper<array_type, false>;
    };

    template<class T, std::size_t N>
    class xmap_tensor : public xsparse_container<xmap_tensor<T, N>>,
                        public xcontainer_semantic<xmap_tensor<T, N>>
    {
    public:

        using self_type = xmap_tensor<T, N>;
        using base_type = xsparse_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
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

        xmap_tensor();
        explicit xmap_tensor(const shape_type& shape);

        ~xmap_tensor() = default;

        xmap_tensor(const xmap_tensor&) = default;
        xmap_tensor& operator=(const xmap_tensor&) = default;

        xmap_tensor(xmap_tensor&&) = default;
        xmap_tensor& operator=(xmap_tensor&&) = default;

        template <class E>
        xmap_tensor(const xexpression<E>& e);

        template <class E>
        self_type& operator=(const xexpression<E>& e);
    };

    /******************************
     * xmap_tensor implementation *
     ******************************/

    template<class T, std::size_t N>
    inline xmap_tensor<T, N>::xmap_tensor()
        : base_type()
    {
    }

    template<class T, std::size_t N>
    inline xmap_tensor<T, N>::xmap_tensor(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }

    template <class T, std::size_t N>
    template <class E>
    inline xmap_tensor<T, N>::xmap_tensor(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    template <class T, size_t N>
    template <class E>
    inline auto xmap_tensor<T, N>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
}

#endif

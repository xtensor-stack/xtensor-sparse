#ifndef XSPARSE_CSF_ARRAY_HPP
#define XSPARSE_CSF_ARRAY_HPP

#include <vector>
#include <xtensor/xiterator.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsemantic.hpp>
#include <xtensor/xstorage.hpp>

#include "xcsf_scheme.hpp"
#include "xsparse_container.hpp"

namespace xt
{
    /**************************
     * xcsf_array declaration *
     **************************/

    template <class T>
    class xcsf_array;

    template<class T>
    struct xcontainer_inner_types<xcsf_array<T>>
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
        using scheme_type = xcsf_scheme<std::vector<svector<size_type>>,
                                        std::vector<svector<size_type>>,
                                        storage_type,
                                        index_type>;

        using reference = xsparse_reference<scheme_type>;
        using temporary_type = xcsf_array<T>;
    };
    
    template <class T>
    struct xiterable_inner_types<xcsf_array<T>>
    {
        using array_type = xcsf_array<T>;
        using inner_shape_type = typename xcontainer_inner_types<array_type>::shape_type;
        using const_stepper = xindexed_stepper<array_type, true>;
        using stepper = xindexed_stepper<array_type, false>;
    };

    template <class T>
    class xcsf_array : public xsparse_container<xcsf_array<T>>,
                       public xcontainer_semantic<xcsf_array<T>>
    {
    public:

        using self_type = xcsf_array<T>;
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

        xcsf_array();
        explicit xcsf_array(const shape_type& shape);

        ~xcsf_array() = default;

        xcsf_array(const xcsf_array&) = default;
        xcsf_array& operator=(const xcsf_array&) = default;

        xcsf_array(xcsf_array&&) = default;
        xcsf_array& operator=(xcsf_array&&) = default;

        template <class E>
        xcsf_array(const xexpression<E>& e);

        template <class E>
        self_type& operator=(const xexpression<E>& e);
    };

    /*****************************
     * xcsf_array implementation *
     *****************************/

    template<class T>
    inline xcsf_array<T>::xcsf_array()
        : base_type()
    {
    }

    template<class T>
    inline xcsf_array<T>::xcsf_array(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }

    template <class T>
    template <class E>
    inline xcsf_array<T>::xcsf_array(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    template <class T>
    template <class E>
    inline auto xcsf_array<T>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
}

#endif

#ifndef XSPARSE_COO_ARRAY_HPP
#define XSPARSE_COO_ARRAY_HPP

#include "xcoo_container.hpp"

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
        using base_type = xcoo_container<xcoo_array<T>>;
        using value_type = T;
        using reference = xsparse_reference<base_type>;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using index_type = svector<size_type>;
        using index_storage_type = std::vector<index_type>;
        using storage_type = std::vector<value_type>;

        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;
        using inner_shape_type = shape_type;
    };
    
    template <class T>
    class xcoo_array : public xcoo_container<xcoo_array<T>> 
    {
    public:

        using self_type = xcoo_array<T>;
        using base_type = xcoo_container<self_type>;
        using storage_type = typename base_type::storage_type;
        using index_storage_type = typename base_type::index_storage_type;
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

    private:

        using strides_type = typename base_type::strides_type;

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xsparse_container_old<xcoo_array<T>>;
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

    template<class T>
    inline auto xcoo_array<T>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template<class T>
    inline auto xcoo_array<T>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }
}

#endif

#ifndef XSPARSE_MAP_ARRAY_HPP
#define XSPARSE_MAP_ARRAY_HPP

#include <map>

#include "xmap_container.hpp"

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
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using index_type = svector<size_type>;
        using storage_type = std::map<index_type, value_type>;

        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;
        using inner_shape_type = shape_type;
    };
    
    template <class T>
    class xmap_array : public xmap_container<xmap_array<T>> 
    {
    public:

        using self_type = xmap_array<T>;
        using base_type = xmap_container<self_type>;
        using storage_type = typename base_type::storage_type;
        using index_type = typename base_type::index_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;

        xmap_array();
        explicit xmap_array(const shape_type& shape);

        ~xmap_array() = default;

        xmap_array(const xmap_array&) = default;
        xmap_array& operator=(const xmap_array&) = default;

        xmap_array(xmap_array&&) = default;
        xmap_array& operator=(xmap_array&&) = default;

    private:

        using strides_type = typename base_type::strides_type;

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xmap_container<xmap_array<T>>;
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

    template<class T>
    inline auto xmap_array<T>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template<class T>
    inline auto xmap_array<T>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }
}

#endif


#ifndef XSPARSE_CSF_ARRAY_HPP
#define XSPARSE_CSF_ARRAY_HPP

#include "xcsf_container.hpp"

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
        using base_type = xcsf_container<xcsf_array<T>>;
        using value_type = T;
        using reference = xsparse_reference<base_type>;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using index_type = svector<size_type>;
        using position_type = std::vector<index_type>;
        using index_storage_type = std::vector<index_type>;
        using storage_type = std::vector<value_type>;

        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;
        using inner_shape_type = shape_type;
    };
    
    template <class T>
    class xcsf_array : public xcsf_container<xcsf_array<T>> 
    {
    public:

        using self_type = xcsf_array<T>;
        using base_type = xcsf_container<self_type>;
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

        xcsf_array();
        explicit xcsf_array(const shape_type& shape);

        ~xcsf_array() = default;

        xcsf_array(const xcsf_array&) = default;
        xcsf_array& operator=(const xcsf_array&) = default;

        xcsf_array(xcsf_array&&) = default;
        xcsf_array& operator=(xcsf_array&&) = default;

    private:

        using strides_type = typename base_type::strides_type;

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xsparse_container<xcsf_array<T>>;
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

    template<class T>
    inline auto xcsf_array<T>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template<class T>
    inline auto xcsf_array<T>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }
}

#endif
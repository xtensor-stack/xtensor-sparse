#ifndef XSPARSE_COO_TENSOR_HPP
#define XSPARSE_COO_TENSOR_HPP

#include "xcoo_container.hpp"

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
        using base_type = xcoo_container<xcoo_tensor<T, N>>;
        using value_type = T;
        using reference = xsparse_reference<base_type>;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using index_type = std::array<size_type, N>;
        using index_storage_type = std::vector<index_type>;
        using storage_type = std::vector<value_type>;

        using shape_type = std::array<size_type, N>;
        using strides_type = std::array<size_type, N>;
        using inner_shape_type = shape_type;
    };
    
    template <class T, std::size_t N>
    class xcoo_tensor : public xcoo_container<xcoo_tensor<T, N>> 
    {
    public:

        using self_type = xcoo_tensor<T, N>;
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

        xcoo_tensor();
        explicit xcoo_tensor(const shape_type& shape);

        ~xcoo_tensor() = default;

        xcoo_tensor(const xcoo_tensor&) = default;
        xcoo_tensor& operator=(const xcoo_tensor&) = default;

        xcoo_tensor(xcoo_tensor&&) = default;
        xcoo_tensor& operator=(xcoo_tensor&&) = default;

    private:

        using strides_type = typename base_type::strides_type;

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xsparse_container_old<xcoo_tensor<T, N>>;
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

    template<class T, std::size_t N>
    inline auto xcoo_tensor<T, N>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template<class T, std::size_t N>
    inline auto xcoo_tensor<T, N>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }
}

#endif

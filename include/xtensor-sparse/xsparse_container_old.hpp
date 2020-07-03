#ifndef XSPARSE_XSPARSE_CONTAINER_OLD_HPP
#define XSPARSE_XSPARSE_CONTAINER_OLD_HPP

#include <xtensor/xstrides.hpp>

#include "xsparse_reference.hpp"

namespace xt
{
    /*************************************
     * xsparse_container_old declaration *
     *************************************/

    template <class D>
    class xsparse_container_old
    {
    public:

        using derived_type = D;

        using inner_types = xcontainer_inner_types<D>;
        using storage_type = typename inner_types::storage_type;
        using index_storage_type = typename inner_types::index_storage_type;
        using index_type = typename inner_types::index_type;
        using value_type = typename storage_type::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename storage_type::pointer;
        using const_pointer = typename storage_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename storage_type::difference_type;

        using shape_type = typename inner_types::shape_type;
        using inner_shape_type = typename inner_types::inner_shape_type;

        size_type size() const noexcept;

        const inner_shape_type& shape() const noexcept;
        size_type dimension() const noexcept;

        template <class S = shape_type>
        void resize(S&& shape, bool force);
        template <class S = shape_type>
        void resize(S&& shape);

        template <class S = shape_type>
        auto& reshape(S&& shape) &;
        template <class T>
        auto& reshape(std::initializer_list<T> shape) &;

        const storage_type& storage() const noexcept;
        storage_type& storage() noexcept;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

    protected:

        using strides_type = typename inner_types::strides_type;

        xsparse_container_old();
        ~xsparse_container_old() = default;

        xsparse_container_old(const xsparse_container_old&) = default;
        xsparse_container_old& operator=(const xsparse_container_old&) = default;

        xsparse_container_old(xsparse_container_old&&) = default;
        xsparse_container_old& operator=(xsparse_container_old&&) = default;

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const & noexcept;
        derived_type derived_cast() && noexcept;

        const strides_type& strides() const noexcept;

    private:


        template <class S = shape_type>
        void reshape_impl(S&& shape, std::false_type);
        template <class S = shape_type>
        void reshape_impl(S&& shape, std::true_type);

        inner_shape_type m_shape;
        strides_type m_strides;

        friend class xsparse_reference<xsparse_container_old<D>>;
    };

    /****************************************
     * xsparse_container_old implementation *
     ****************************************/

    template<class D>
    xsparse_container_old<D>::xsparse_container_old()
    {
        m_shape = xtl::make_sequence<inner_shape_type>(dimension(), 0);
        m_strides = xtl::make_sequence<strides_type>(dimension(), 0);
    }

    template <class D>
    inline auto xsparse_container_old<D>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    template <class D>
    inline auto xsparse_container_old<D>::dimension() const noexcept -> size_type
    {
        return shape().size();
    }

    template <class D>
    inline auto xsparse_container_old<D>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class D>
    inline auto xsparse_container_old<D>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    template <class D>
    template <class S>
    inline void xsparse_container_old<D>::resize(S&& shape, bool force)
    {
        std::size_t dim = shape.size();
        if (m_shape.size() != dim || !std::equal(std::begin(shape), std::end(shape), std::begin(m_shape)) || force)
        {
            m_shape = xtl::forward_sequence<shape_type, S>(shape);
            strides_type old_strides = strides();
            resize_container(m_strides, dim);
            compute_strides(m_shape, XTENSOR_DEFAULT_LAYOUT, m_strides);
            derived_cast().update_entries(old_strides);
        }
    }

    template <class D>
    template <class S>
    inline void xsparse_container_old<D>::resize(S&& shape)
    {
        resize(std::forward<S>(shape), true);
    }

    template <class D>
    template <class S>
    inline auto& xsparse_container_old<D>::reshape(S&& shape) &
    {
        reshape_impl(std::forward<S>(shape), std::is_signed<std::decay_t<typename std::decay_t<S>::value_type>>());
        return *this;
    }

    template <class D>
    template <class T>
    inline auto& xsparse_container_old<D>::reshape(std::initializer_list<T> shape) &
    {
        using sh_type = rebind_container_t<D, shape_type>;
        sh_type sh = xtl::make_sequence<sh_type>(shape.size());
        std::copy(shape.begin(), shape.end(), sh.begin());
        reshape_impl(std::move(sh), std::is_signed<T>());
        return *this;
    }
    
    template <class D>
    template <class S>
    inline void xsparse_container_old<D>::reshape_impl(S&& shape, std::false_type /* is unsigned */)
    {
        if (compute_size(shape) != this->size())
        {
            XTENSOR_THROW(std::runtime_error, "Cannot reshape with incorrect number of elements. Do you mean to resize?");
        }

        std::size_t dim = shape.size();
        strides_type old_strides = strides();
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        resize_container(m_strides, dim);
        compute_strides(m_shape, XTENSOR_DEFAULT_LAYOUT, m_strides);
        derived_cast().update_entries(old_strides);
    }

    template <class D>
    template <class S>
    inline void xsparse_container_old<D>::reshape_impl(S&& _shape, std::true_type /* is signed */)
    {
        using value_type = typename std::decay_t<S>::value_type;
        auto new_size = compute_size(_shape);
        if (this->size() % new_size)
        {
            XTENSOR_THROW(std::runtime_error, "Negative axis size cannot be inferred. Shape mismatch.");
        }
        std::decay_t<S> shape = _shape;
        value_type accumulator = 1;
        std::size_t neg_idx = 0;
        std::size_t i = 0;
        for(auto it = shape.begin(); it != shape.end(); ++it, i++)
        {
            auto&& dim = *it;
            if(dim < 0)
            {
                XTENSOR_ASSERT(dim == -1 && !neg_idx);
                neg_idx = i;
            }
            accumulator *= dim;
        }
        if(accumulator < 0)
        {
            shape[neg_idx] = static_cast<value_type>(this->size()) / std::abs(accumulator);
        }
        else if(this->size() != new_size)
        {
            XTENSOR_THROW(std::runtime_error, "Cannot reshape with incorrect number of elements. Do you mean to resize?");
        }
        
        std::size_t dim = shape.size();
        auto old_strides = strides();
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        resize_container(m_strides, dim);
        compute_strides(m_shape, XTENSOR_DEFAULT_LAYOUT, m_strides);
        derived_cast().update_entries(old_strides);
    }

    template <class D>
    inline auto xsparse_container_old<D>::storage() const noexcept -> const storage_type&
    {
        return derived_cast().storage_impl();
    }

    template <class D>
    inline auto xsparse_container_old<D>::storage() noexcept -> storage_type&
    {
        return derived_cast().storage_impl();
    }

    template<class D>
    template<class... Args>
    inline auto xsparse_container_old<D>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return derived_cast().access_impl(args...);
    }

    template<class D>
    template<class... Args>
    inline auto xsparse_container_old<D>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return derived_cast().access_impl(args...);
    }

    template <class D>
    inline auto xsparse_container_old<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xsparse_container_old<D>::derived_cast() const & noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline auto xsparse_container_old<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }
}

#endif

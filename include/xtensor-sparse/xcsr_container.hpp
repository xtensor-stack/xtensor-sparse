#ifndef XSPARSE_CSR_CONTAINER_HPP
#define XSPARSE_CSR_CONTAINER_HPP

#include <array>
#include <utility>
#include <vector>

#include <xtensor/xstorage.hpp>
#include <xtensor/xexception.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace xt
{
    template<class T,>
    class xcsr_storage
    {
    public:

        

    private:


    }

    /******************************
     * xcsr_container declaration *
     ******************************/

    template<class D>
    class xcsr_container
    {
    public:

        using derived_type = D;

        using inner_types = xcontainer_inner_types<D>;
        using storage_type = typename inner_types::storage_type;
        using index_type = typename inner_types::index_type;
        using value_type = typename storage_type::mapped_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename storage_type::pointer;
        using const_pointer = typename storage_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename storage_type::difference_type;

        using shape_type = typename inner_types::shape_type;
        using inner_shape_type = typename inner_types::inner_shape_type;

        size_type size() const noexcept;

        XTENSOR_CONSTEXPR_RETURN const inner_shape_type& shape() const noexcept;
        XTENSOR_CONSTEXPR_RETURN size_type dimension() const noexcept;

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

        xcsr_container();
        ~xcsr_container() = default;

        xcsr_container(const xcsr_container&) = default;
        xcsr_container& operator=(const xcsr_container&) = default;

        xcsr_container(xcsr_container&&) = default;
        xcsr_container& operator=(xcsr_container&&) = default;

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const & noexcept;
        derived_type derived_cast() && noexcept;

    private:

        static const value_type ZERO;

        XTENSOR_CONSTEXPR_RETURN const strides_type& strides() const noexcept;

        template<class... Args>
        const_reference access_impl(Args... args) const;
        template<class... Args>
        reference access_impl(Args... args);

        template <class S = shape_type>
        void reshape_impl(S&& shape, std::false_type);
        template <class S = shape_type>
        void reshape_impl(S&& shape, std::true_type);
        void update_entries(const strides_type& old_strides);

        inner_shape_type m_shape;
        strides_type m_strides;
    };

    /*********************************
     * xcsr_container implementation *
     *********************************/

    template<class D>
    const typename xcsr_container<D>::value_type xcsr_container<D>::ZERO = 0;

    template<class D>
    xcsr_container<D>::xcsr_container()
    {
        m_shape = xtl::make_sequence<inner_shape_type>(dimension(), 0);
        m_strides = xtl::make_sequence<strides_type>(dimension(), 0);
    }

    template <class D>
    inline auto xcsr_container<D>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    template <class D>
    XTENSOR_CONSTEXPR_RETURN auto xcsr_container<D>::dimension() const noexcept -> size_type
    {
        return shape().size();
    }

    template <class D>
    XTENSOR_CONSTEXPR_RETURN auto xcsr_container<D>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class D>
    XTENSOR_CONSTEXPR_RETURN auto xcsr_container<D>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    template <class D>
    template <class S>
    inline void xcsr_container<D>::resize(S&& shape, bool force)
    {
        std::size_t dim = shape.size();
        if (m_shape.size() != dim || !std::equal(std::begin(shape), std::end(shape), std::begin(m_shape)) || force)
        {
            m_shape = xtl::forward_sequence<shape_type, S>(shape);
            strides_type old_strides = strides();
            resize_container(m_strides, dim);
            compute_strides(m_shape, XTENSOR_DEFAULT_LAYOUT, m_strides);
            update_entries(old_strides);
        }
    }

    template <class D>
    template <class S>
    inline void xcsr_container<D>::resize(S&& shape)
    {
        resize(std::forward<S>(shape), true);
    }

    template <class D>
    template <class S>
    inline auto& xcsr_container<D>::reshape(S&& shape) &
    {
        reshape_impl(std::forward<S>(shape), std::is_signed<std::decay_t<typename std::decay_t<S>::value_type>>());
        return *this;
    }

    template <class D>
    template <class T>
    inline auto& xcsr_container<D>::reshape(std::initializer_list<T> shape) &
    {
        using sh_type = rebind_container_t<D, shape_type>;
        sh_type sh = xtl::make_sequence<sh_type>(shape.size());
        std::copy(shape.begin(), shape.end(), sh.begin());
        reshape_impl(std::move(sh), std::is_signed<T>());
        return *this;
    }
    
    template <class D>
    template <class S>
    inline void xcsr_container<D>::reshape_impl(S&& shape, std::false_type /* is unsigned */)
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
        update_entries(old_strides);
    }

    template <class D>
    template <class S>
    inline void xcsr_container<D>::reshape_impl(S&& _shape, std::true_type /* is signed */)
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
        update_entries(old_strides);
    }

    template <class D>
    inline auto xcsr_container<D>::storage() const noexcept -> const storage_type&
    {
        return derived_cast().storage_impl();
    }

    template <class D>
    inline auto xcsr_container<D>::storage() noexcept -> storage_type&
    {
        return derived_cast().storage_impl();
    }

    template<class D>
    template<class... Args>
    inline auto xcsr_container<D>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(args...);
    }

    template<class D>
    template<class... Args>
    inline auto xcsr_container<D>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(args...);
    }

    template<class D>
    template<class... Args>
    inline auto xcsr_container<D>::access_impl(Args... args) const -> const_reference
    {
        // TODO: check if all args have the good type
        index_type key{static_cast<size_type>(args)...};

        auto it = storage().find(key);
        if (it == storage().end())
        {
            return ZERO;
        }
        return storage()[key];
    }

    template<class D>
    template<class... Args>
    inline auto xcsr_container<D>::access_impl(Args... args) -> reference
    {
        // TODO: check if all args have the good type
        index_type key{static_cast<size_type>(args)...};

        auto it = storage().find(key);
        if (it == storage().end())
        {
            storage()[key] = ZERO;
        }
        return storage()[key];
    }

    template <class D>
    inline void xcsr_container<D>::update_entries(const strides_type& old_strides)
    {
        storage_type new_data;
        for(auto& c: storage())
        {
            auto& old_key = c.first;
            size_type index = element_offset<size_type>(old_strides, old_key.cbegin(), old_key.cend());
            shape_type new_key = unravel_from_strides(index, strides());
            new_data[new_key] = c.second;
        }

        std::swap(storage(), new_data);
    }

    template <class D>
    inline auto xcsr_container<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xcsr_container<D>::derived_cast() const & noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline auto xcsr_container<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    /**************************
     * xcsr_array declaration *
     **************************/

    template<class T>
    class xcsr_array: public xcsr_container<xcsr_array<T>> 
    {
    public:

        using self_type = xcsr_array<T>;
        using base_type = xcsr_container<self_type>;
        using storage_type = typename base_type::storage_type;
        using index_type = typename base_type::index_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;

        xcsr_array();
        explicit xcsr_array(const shape_type& shape);

        ~xcsr_array() = default;

        xcsr_array(const xcsr_array&) = default;
        xcsr_array& operator=(const xcsr_array&) = default;

        xcsr_array(xcsr_array&&) = default;
        xcsr_array& operator=(xcsr_array&&) = default;

    private:

        using strides_type = typename base_type::strides_type;

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xcsr_container<xcsr_array<T>>;
    };

    /*****************************
     * xcsr_array implementation *
     *****************************/
    template<class T>
    inline xcsr_array<T>::xcsr_array()
        : base_type()
    {}

    template<class T>
    inline xcsr_array<T>::xcsr_array(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }

    template<class T>
    inline auto xcsr_array<T>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template<class T>
    inline auto xcsr_array<T>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template<class T>
    struct xcontainer_inner_types<xcsr_array<T>>
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

    /***************************
     * xcsr_tensor declaration *
     ***************************/

    template<class T, std::size_t N>
    class xcsr_tensor: public xcsr_container<xcsr_tensor<T, N>> 
    {
    public:

        using self_type = xcsr_tensor<T, N>;
        using base_type = xcsr_container<self_type>;
        using storage_type = typename base_type::storage_type;
        using index_type = typename base_type::index_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;

        xcsr_tensor();
        explicit xcsr_tensor(const shape_type& shape);

        ~xcsr_tensor() = default;

        xcsr_tensor(const xcsr_tensor&) = default;
        xcsr_tensor& operator=(const xcsr_tensor&) = default;

        xcsr_tensor(xcsr_tensor&&) = default;
        xcsr_tensor& operator=(xcsr_tensor&&) = default;

    private:

        using strides_type = typename base_type::strides_type;

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xcsr_container<xcsr_tensor<T, N>>;
    };

    /******************************
     * xcsr_tensor implementation *
     ******************************/

    template<class T, std::size_t N>
    inline xcsr_tensor<T, N>::xcsr_tensor()
        : base_type()
    {}

    template<class T, std::size_t N>
    inline xcsr_tensor<T, N>::xcsr_tensor(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }

    template<class T, std::size_t N>
    inline auto xcsr_tensor<T, N>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template<class T, std::size_t N>
    inline auto xcsr_tensor<T, N>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template<class T, std::size_t N>
    struct xcontainer_inner_types<xcsr_tensor<T, N>>
    {
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using index_type = std::array<size_type, N>;
        using storage_type = std::map<index_type, value_type>;

        using shape_type = std::array<size_type, N>;
        using strides_type = std::array<size_type, N>;
        using inner_shape_type = shape_type;
    };
}
#endif


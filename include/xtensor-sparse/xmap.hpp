#ifndef XTENSOR_SPARSE_LIL_HPP
#define XTENSOR_SPARSE_LIL_HPP

#include <array>
#include <map>
#include <utility>
#include <vector>

#include <xtensor/xstorage.hpp>
#include <xtensor/xexception.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace xt
{

    template <class shape_type, class strides_type>
    inline void compute_strides(const shape_type& shape, strides_type& strides)
    {
        using strides_value_type = typename std::decay_t<strides_type>::value_type;
        strides_value_type data_size = 1;
        for (std::size_t i = shape.size(); i != 0; --i)
        {
            strides[i - 1] = data_size;
            data_size = strides[i - 1] * static_cast<strides_value_type>(shape[i - 1]);
            if (shape[i - 1] == 1)
            {
                strides[i - 1] = 0;
            }
        }
    }

    template<class T>
    class xmap_container
    {
    public:

        using value_type = T;
        using const_reference = const T&;
        using reference = T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;

        using inner_shape_type = svector<size_type>;

        xmap_container(const shape_type& shape);

        size_type size() const noexcept;

        const shape_type& shape() const;
        size_type dimension() const;
        const strides_type& strides() const;
        
        template <class S = shape_type>
        void resize(S&& shape, bool force);
        template <class S = shape_type>
        void resize(S&& shape);

        template <class S = shape_type>
        auto& reshape(S&& shape) &;
        template <class TT>
        auto& reshape(std::initializer_list<TT> shape) &;

        template<class... Args>
        const_reference operator()(Args... args) const;
        template<class... Args>
        reference operator()(Args... args);

    private:

        using index_type = svector<size_type>;
        using container_type = std::map<index_type, value_type>;

        // FIXME: set constexpr
        static const value_type ZERO;

        template<class... Args>
        const_reference access_impl(Args... args) const;
        template<class... Args>
        reference access_impl(Args... args);

        template <class S = shape_type>
        void reshape_impl(S&& shape, std::false_type);
        template <class S = shape_type>
        void reshape_impl(S&& shape, std::true_type);
        void update_keys(const strides_type& old_strides);
        
        shape_type m_shape;
        strides_type m_strides;
        container_type m_data;
    };

    template<class T>
    const typename xmap_container<T>::value_type xmap_container<T>::ZERO = 0;

    template<class T>
    inline xmap_container<T>::xmap_container(const shape_type& shape)
    {
        resize(shape);
    }

    template <class T>
    inline auto xmap_container<T>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    template<class T>
    inline auto xmap_container<T>::shape() const -> const shape_type&
    {
        return m_shape;
    }

    template<class T>
    inline auto xmap_container<T>::strides() const -> const strides_type&
    {
        return m_strides;
    }

    template<class T>
    inline auto xmap_container<T>::dimension() const -> size_type
    {
        return m_shape.size();
    }

    template <class T>
    template <class S>
    inline void xmap_container<T>::resize(S&& shape, bool force)
    {
        std::size_t dim = shape.size();
        if (m_shape.size() != dim || !std::equal(std::begin(shape), std::end(shape), std::begin(m_shape)) || force)
        {
            m_shape = xtl::forward_sequence<shape_type, S>(shape);
            strides_type old_strides = strides();
            resize_container(m_strides, dim);
            compute_strides(m_shape, m_strides);
            update_keys(old_strides);
        }
    }

    template <class T>
    template <class S>
    inline void xmap_container<T>::resize(S&& shape)
    {
        resize(std::forward<S>(shape), true);
    }

    template <class T>
    template <class S>
    inline auto& xmap_container<T>::reshape(S&& shape) &
    {
        reshape_impl(std::forward<S>(shape), std::is_signed<std::decay_t<typename std::decay_t<S>::value_type>>());
        return *this;
    }

    template <class T>
    template <class TT>
    inline auto& xmap_container<T>::reshape(std::initializer_list<TT> shape) &
    {
        using sh_type = rebind_container_t<T, shape_type>;
        sh_type sh = xtl::make_sequence<sh_type>(shape.size());
        std::copy(shape.begin(), shape.end(), sh.begin());
        reshape_impl(std::move(sh), std::is_signed<TT>());
        return *this;
    }
    
    template <class T>
    template <class S>
    inline void xmap_container<T>::reshape_impl(S&& shape, std::false_type /* is unsigned */)
    {
        if (compute_size(shape) != this->size())
        {
            XTENSOR_THROW(std::runtime_error, "Cannot reshape with incorrect number of elements. Do you mean to resize?");
        }

        std::size_t dim = shape.size();
        strides_type old_strides = strides();
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        resize_container(m_strides, dim);
        compute_strides(m_shape, m_strides);
        update_keys(old_strides);
    }

    template <class T>
    template <class S>
    inline void xmap_container<T>::reshape_impl(S&& _shape, std::true_type /* is signed */)
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
        compute_strides(m_shape, m_strides);
        update_keys(old_strides);
    }

    template <class T>
    inline void xmap_container<T>::update_keys(const strides_type& old_strides)
    {
        container_type new_data;
        for(auto& c: m_data)
        {
            auto& old_key = c.first;
            size_type index = element_offset<size_type>(old_strides, old_key.cbegin(), old_key.cend());
            shape_type new_key = unravel_from_strides(index, strides());
            new_data[new_key] = c.second;
        }

        std::swap(m_data, new_data);
    }

    template<class T>
    template<class... Args>
    inline auto xmap_container<T>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(args...);
    }

    template<class T>
    template<class... Args>
    inline auto xmap_container<T>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(args...);
    }

    template<class T>
    template<class... Args>
    inline auto xmap_container<T>::access_impl(Args... args) const -> const_reference
    {
        // TODO: check if all args have the good type
        index_type key{static_cast<size_type>(args)...};

        auto it = m_data.find(key);
        if (it == m_data.end())
        {
            return ZERO;
        }
        return m_data[key];
    }

    template<class T>
    template<class... Args>
    inline auto xmap_container<T>::access_impl(Args... args) -> reference
    {
        // TODO: check if all args have the good type
        index_type key{static_cast<size_type>(args)...};

        auto it = m_data.find(key);
        if (it == m_data.end())
        {
            m_data[key] = ZERO;
        }
        return m_data[key];
    }
}
#endif


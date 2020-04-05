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
    template<class T>
    class xlil_container
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
        using backstrides_type = svector<size_type>;

        using inner_shape_type = svector<size_type>;
        using inner_strides_type = svector<size_type>;
        using inner_backstrides_type = svector<size_type>;

        xlil_container(const shape_type& shape);

        size_type size() const noexcept;

        const shape_type& shape() const;
        size_type dimension() const;
        
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
        template <class S = shape_type>
        void update_keys(S&& shape);
        
        shape_type m_shape;
        container_type m_data;
    };

    template<class T>
    const typename xlil_container<T>::value_type xlil_container<T>::ZERO = 0;

    template<class T>
    inline xlil_container<T>::xlil_container(const shape_type& shape)
     : m_shape{shape}
    {}

    template <class T>
    inline auto xlil_container<T>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    template<class T>
    inline auto xlil_container<T>::shape() const -> const shape_type&
    {
        return m_shape;
    }

    template<class T>
    inline auto xlil_container<T>::dimension() const -> size_type
    {
        return m_shape.size();
    }

    template <class T>
    template <class S>
    inline void xlil_container<T>::resize(S&& shape, bool force)
    {
        std::size_t dim = shape.size();
        if (m_shape.size() != dim || !std::equal(std::begin(shape), std::end(shape), std::begin(m_shape)) || force)
        {
            m_shape = xtl::forward_sequence<shape_type, S>(shape);
        }
    }

    template <class T>
    template <class S>
    inline void xlil_container<T>::resize(S&& shape)
    {
        resize(std::forward<S>(shape), true);
    }

    template <class T>
    template <class S>
    inline auto& xlil_container<T>::reshape(S&& shape) &
    {
        reshape_impl(std::forward<S>(shape), std::is_signed<std::decay_t<typename std::decay_t<S>::value_type>>());
        return *this;
    }

    template <class T>
    template <class TT>
    inline auto& xlil_container<T>::reshape(std::initializer_list<TT> shape) &
    {
        using sh_type = rebind_container_t<T, shape_type>;
        sh_type sh = xtl::make_sequence<sh_type>(shape.size());
        std::copy(shape.begin(), shape.end(), sh.begin());
        reshape_impl(std::move(sh), std::is_signed<TT>());
        return *this;
    }
    
    template <class T>
    template <class S>
    inline void xlil_container<T>::reshape_impl(S&& shape, std::false_type /* is unsigned */)
    {
        if (compute_size(shape) != this->size())
        {
            XTENSOR_THROW(std::runtime_error, "Cannot reshape with incorrect number of elements. Do you mean to resize?");
        }

        update_keys(std::forward<S>(shape));
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
    }

    template <class T>
    template <class S>
    inline void xlil_container<T>::reshape_impl(S&& _shape, std::true_type /* is signed */)
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
        
        update_keys(shape);
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
    }

    template <class T>
    template <class S>
    inline void xlil_container<T>::update_keys(S&& shape)
    {
        container_type new_data;
        for(auto& c: m_data)
        {
            auto& old_key = c.first;

            // compute the offset for the old_key
            size_type index = 0;
            size_type accumulator = 1;
            for (std::size_t i = m_shape.size(); i != 0; --i)
            {
                index += old_key[i - 1]*accumulator;
                accumulator *= m_shape[i - 1];
            }

            // compute the new key from the offset and the new shape
            index_type new_key = xtl::make_sequence<index_type>(shape.size());
            accumulator = 1;
            for(std::size_t i = 0; i < shape.size(); ++i)
            {
                accumulator *= static_cast<size_type>(shape[i]);
            }
            
            for(std::size_t i = 0; i < shape.size(); ++i)
            {
                accumulator /= static_cast<size_type>(shape[i]);
                new_key[i] = index/accumulator;
                index -= new_key[i]*accumulator;
            }

            new_data[new_key] = c.second;
        }

        std::swap(m_data, new_data);
    }

    template<class T>
    template<class... Args>
    inline auto xlil_container<T>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(args...);
    }

    template<class T>
    template<class... Args>
    inline auto xlil_container<T>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(args...);
    }

    template<class T>
    template<class... Args>
    inline auto xlil_container<T>::access_impl(Args... args) const -> const_reference
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
    inline auto xlil_container<T>::access_impl(Args... args) -> reference
    {
        // TODO: check if all args have the good type
        index_type key{static_cast<size_type>(args)...};

        auto it = m_data.find(key);
        if (it == m_data.end())
            m_data[key] = ZERO;
        }
            return m_data[key];
    }
}
#endif


#ifndef XTENSOR_SPARSE_LIL_HPP
#define XTENSOR_SPARSE_LIL_HPP

#include <array>
#include <map>
#include <utility>
#include <vector>

#include <xtensor/xstorage.hpp>
#include <xtensor/xexception.hpp>
#include <xtensor/xtensor.hpp>
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

        using inner_shape_type = svector<size_type>;

        xlil_container(const shape_type& shape);

        const shape_type& shape() const;
        size_type dimension() const;

        template<class... Args>
        const_reference operator()(Args... args) const;
        template<class... Args>
        reference operator()(Args... args);

    private:

        using index_type = size_type;
        using list_index_type = svector<size_type>;
        using container_type = std::map<list_index_type, value_type>;

        // FIXME: set constexpr
        static const value_type ZERO;

        template<class... Args>
        const_reference access_impl(Args... args) const;
        template<class... Args>
        reference access_impl(Args... args);

        shape_type m_shape;
        container_type m_data;
    };

    template<class T>
    const typename xlil_container<T>::value_type xlil_container<T>::ZERO = 0;

    template<class T>
    inline xlil_container<T>::xlil_container(const shape_type& shape)
     : m_shape{shape}
    {}

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
        list_index_type key{args...};

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
        list_index_type key{args...};

        auto it = m_data.find(key);
        if (it == m_data.end())
        {
            m_data[key] = ZERO;
        }
            return m_data[key];
    }
}
#endif


#ifndef XTENSOR_SPARSE_LIL_HPP
#define XTENSOR_SPARSE_LIL_HPP

#include <array>
#include <map>
#include <utility>
#include <vector>

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

        template<class... Args>
        const_reference operator()(Args... args) const;
        template<class... Args>
        reference operator()(Args... args);

    private:

        using index_type = size_type;
        using list_index_type = std::vector<index_type>;
        using container_type = std::map<list_index_type, std::map<index_type, value_type>>;

        // FIXME: set constexpr
        static const value_type ZERO;

        template<class... Args>
        const_reference access_impl(Args... args) const;
        template<class... Args>
        reference access_impl(Args... args);

        template<class... Args>
        std::pair<list_index_type, index_type> build_keys(Args... args) const;

        container_type m_data;
    };

    template<class T>
    const typename xlil_container<T>::value_type xlil_container<T>::ZERO = 0;

    template<class T>
    template<class... Args>
    auto xlil_container<T>::operator()(Args... args) const -> const_reference
    {
        return access_impl(args...);
    }

    template<class T>
    template<class... Args>
    auto xlil_container<T>::operator()(Args... args) -> reference
    {
        return access_impl(args...);
    }

    template<class T>
    template<class... Args>
    auto xlil_container<T>::access_impl(Args... args) const -> const_reference
    {
        list_index_type key1;
        index_type key2;

        std::tie(key1, key2) = build_keys(args...);

        auto it = m_data.find(key1);
        if (it == m_data.end())
        {
            return ZERO;
        }
        else
        {
            auto it2 = it->find(key2);
            if (it2 == it->end())
            {
                return ZERO;
            }
            return *it2;
        }
    }

    template<class T>
    template<class... Args>
    auto xlil_container<T>::access_impl(Args... args) -> reference
    {
        list_index_type key1;
        index_type key2;

        std::tie(key1, key2) = build_keys(args...);

        auto it = m_data.find(key1);
        if (it == m_data.end())
        {
            m_data[key1][key2] = ZERO;
            return m_data[key1][key2];
        }
        else
        {
            auto it2 = it->second.find(key2);
            if (it2 == it->second.end())
            {
                m_data[key1][key2] = ZERO;
                return m_data[key1][key2];
            }
            return it2->second;
        }
    }

    template<class T>
    template<class... Args>
    auto xlil_container<T>::build_keys(Args... args) const -> std::pair<list_index_type, index_type>
    {
        list_index_type res1;
        index_type res2;

        std::array<int, sizeof...(Args) + 1> {(res1.push_back(args), 0)..., 0};

        res2 = res1.back();
        res1.pop_back();
        return {res1, res2};
    }
}
#endif


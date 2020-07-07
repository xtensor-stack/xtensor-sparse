#ifndef XSPARSE_COO_SCHEME_HPP
#define XSPARSE_COO_SCHEME_HPP

#include <algorithm>

#include <xtensor/xstorage.hpp>
#include <xtensor/xstrides.hpp>

namespace xt
{
    template <class scheme>
    class xcoo_scheme_iterator;

    /***************
     * xcoo_scheme *
     ***************/

    template <class P, class C, class ST, class IT = svector<std::size_t>>
    class xcoo_scheme
    {
    public:

        using self_type = xcoo_scheme<P, C, ST, IT>;
        using position_type = P;
        using coordinate_type = C;
        using storage_type = ST;
        using index_type = IT;

        using value_type = typename storage_type::value_type;
        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;

        const position_type& position() const;
        const coordinate_type& coordinate() const;
        const storage_type& storage() const;

        using iterator = xcoo_scheme_iterator<self_type>;
        using const_iterator = xcoo_scheme_iterator<const self_type>;

        xcoo_scheme();

        pointer find_element(const index_type& index);
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        template<class strides_type, class shape_type>
        void update_entries(const strides_type& old_strides,
                            const strides_type& new_strides,
                            const shape_type& new_shape);

        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

    private:

        position_type m_pos;
        coordinate_type m_coords;
        storage_type m_storage;

        friend class xcoo_scheme_iterator<self_type>;
        friend class xcoo_scheme_iterator<const self_type>;
    };

    /************************
     * xcoo_scheme_iterator *
     ************************/

    namespace detail
    {
        template <class scheme>
        struct xcoo_scheme_storage_type
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::iterator;
        };

        template <class scheme>
        struct xcoo_scheme_storage_type<const scheme>
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::const_iterator;
        };

        template <class scheme>
        struct xcoo_scheme_iterator_types : xcoo_scheme_storage_type<scheme>
        {
            using base_type = xcoo_scheme_storage_type<scheme>;
            using index_type = typename scheme::index_type;
            using coordinate_type = typename scheme::coordinate_type;
            using coordinate_iterator = typename coordinate_type::const_iterator;
            using value_iterator = typename base_type::value_iterator;
            using value_type = typename value_iterator::value_type;
            using reference = typename value_iterator::reference;
            using pointer = typename value_iterator::pointer;
            using difference_type = typename value_iterator::difference_type;
        };
    }

    template <class scheme>
    class xcoo_scheme_iterator
    {
    public:

        using self_type = xcoo_scheme_iterator<scheme>;
        using scheme_type = scheme;
        using iterator_types = detail::xcoo_scheme_iterator_types<scheme>;
        using index_type = typename iterator_types::index_type;
        using coordinate_type = typename iterator_types::coordinate_type;
        using coordinate_iterator = typename iterator_types::coordinate_iterator;
        using value_iterator = typename iterator_types::value_iterator;
        using value_type = typename iterator_types::value_type;
        using reference = typename iterator_types::reference;
        using pointer = typename iterator_types::pointer;
        using difference_type = typename iterator_types::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        xcoo_scheme_iterator();
        xcoo_scheme_iterator(scheme& s, coordinate_iterator cit, value_iterator vit);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;
        const index_type& index() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

    private:

        scheme_type* p_scheme;
        coordinate_iterator m_cit;
        value_iterator m_vit;
    };

    template <class S>
    bool operator==(const xcoo_scheme_iterator<S>& lhs,
                    const xcoo_scheme_iterator<S>& rhs);

    template <class S>
    bool operator<(const xcoo_scheme_iterator<S>& lhs,
                   const xcoo_scheme_iterator<S>& rhs);

    /******************************
     * xcoo_scheme implementation *
     ******************************/

    template <class P, class C, class ST, class IT>
    inline xcoo_scheme<P, C, ST, IT>::xcoo_scheme()
        : m_pos(P{{0u, 0u}})
    {
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::position() const -> const position_type&
    {
        return m_pos;
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::coordinate() const -> const coordinate_type&
    {
        return m_coords;
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::storage() const -> const storage_type&
    {
        return m_storage;
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::find_element(const index_type& index) -> pointer
    {
        auto it = std::find(m_coords.begin(), m_coords.end(), index);
        return it == m_coords.end() ? nullptr : &*(m_storage.begin() + (it - m_coords.begin()));
    }

    template <class P, class C, class ST, class IT>
    inline void xcoo_scheme<P, C, ST, IT>::insert_element(const index_type& index, const_reference value)
    {
        auto it = std::upper_bound(m_coords.cbegin(), m_coords.cend(), index);
        if (it != m_coords.cend())
        {
            auto diff = std::distance(m_coords.cbegin(), it);
            m_coords.insert(it, index);
            m_storage.insert(m_storage.cbegin() + diff, value);
        }
        else
        {
            m_coords.push_back(index);
            m_storage.push_back(value);
        }
        ++m_pos.back();
    }

    template <class P, class C, class ST, class IT>
    inline void xcoo_scheme<P, C, ST, IT>::remove_element(const index_type& index)
    {
        auto it = std::find(m_coords.begin(), m_coords.end(), index);
        if (it != m_coords.end())
        {
            auto diff = it - m_coords.begin();
            m_coords.erase(it);
            m_pos.back()--;
            m_storage.erase(m_storage.begin() + diff);
        }
    }

    template <class P, class C, class ST, class IT>
    template<class strides_type, class shape_type>
    inline void xcoo_scheme<P, C, ST, IT>::update_entries(const strides_type& old_strides,
                                                          const strides_type& new_strides,
                                                          const shape_type&)
    {
        coordinate_type new_coords;

        for(auto& old_index: m_coords)
        {
            std::size_t offset = element_offset<std::size_t>(old_strides, old_index.cbegin(), old_index.cend());
            index_type new_index = unravel_from_strides(offset, new_strides);
            new_coords.push_back(new_index);
        }
        using std::swap;
        swap(m_coords, new_coords);
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::begin() -> iterator
    {
        return iterator(*this, m_coords.cbegin(), m_storage.begin());
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::end() -> iterator
    {
        return iterator(*this, m_coords.cend(), m_storage.end());
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::end() const -> const_iterator
    {
        return cend();
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::cbegin() const -> const_iterator
    {
        return const_iterator(*this, m_coords.cbegin(), m_storage.cbegin());
    }

    template <class P, class C, class ST, class IT>
    inline auto xcoo_scheme<P, C, ST, IT>::cend() const -> const_iterator
    {
        return const_iterator(*this, m_coords.cend(), m_storage.cend());
    }

    /***************************************
     * xcoo_scheme_iterator implementation *
     ***************************************/

    template <class S>
    inline xcoo_scheme_iterator<S>::xcoo_scheme_iterator()
        : p_scheme(nullptr)
    {
    }

    template <class S>
    inline xcoo_scheme_iterator<S>::xcoo_scheme_iterator(S& s,
                                                         coordinate_iterator cit,
                                                         value_iterator vit)
        : p_scheme(&s)
        , m_cit(cit)
        , m_vit(vit)
    {
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::operator++() -> self_type&
    {
        ++m_cit;
        ++m_vit;
        return *this;
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::operator--() -> self_type&
    {
        --m_cit;
        --m_vit;
        return *this;
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::operator+=(difference_type n) -> self_type&
    {
        m_cit += n;
        m_vit += n;
        return *this;
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::operator-=(difference_type n) -> self_type&
    {
        m_cit -= n;
        m_vit -= n;
        return *this;
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_cit - rhs.m_cit;
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::operator*() const -> reference
    {
        return *m_vit;
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::operator->() const -> pointer
    {
        return &(*m_vit);
    }

    template <class S>
    inline auto xcoo_scheme_iterator<S>::index() const -> const index_type&
    {
        return *m_cit;
    }

    template <class S>
    inline bool xcoo_scheme_iterator<S>::equal(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_cit == rhs.m_cit && m_vit == rhs.m_vit;
    }

    template <class S>
    inline bool xcoo_scheme_iterator<S>::less_than(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_cit < rhs.m_cit && m_vit < rhs.m_vit;
    }

    template <class S>
    inline bool operator==(const xcoo_scheme_iterator<S>& lhs,
                           const xcoo_scheme_iterator<S>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class S>
    inline bool operator<(const xcoo_scheme_iterator<S>& lhs,
                          const xcoo_scheme_iterator<S>& rhs)
    {
        return lhs.less_than(rhs);
    }
}

#endif

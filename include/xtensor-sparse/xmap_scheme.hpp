#ifndef XSPARSE_MAP_SCHEME_HPP
#define XSPARSE_MAP_SCHEME_HPP

#include <xtl/xiterator_base.hpp>
#include <xtensor/xstrides.hpp>

namespace xt
{
    template <class scheme>
    class xmap_scheme_nz_iterator;
    
    /***************
     * xmap_scheme *
     ***************/

    template <class ST>
    class xmap_scheme
    {
    public:

        using self_type = xmap_scheme<ST>;
        using storage_type = ST;
        using index_type = typename storage_type::key_type;
        using value_type = typename storage_type::mapped_type;
        using refernece = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        using nz_iterator = xmap_scheme_nz_iterator<self_type>;
        using const_nz_iterator = xmap_scheme_nz_iterator<const self_type>;

        const storage_type& storage() const;

        pointer find_element(const index_type& index);
        const_pointer find_element(const index_type& index) const;
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        template <class strides_type, class shape_type>
        void update_entries(const strides_type& old_strides,
                            const strides_type& new_strides,
                            const shape_type& new_shape);

        nz_iterator nz_begin();
        nz_iterator nz_end();
        const_nz_iterator nz_begin() const;
        const_nz_iterator nz_end() const;
        const_nz_iterator nz_cbegin() const;
        const_nz_iterator nz_cend() const;

    private:

        const_pointer find_element_impl(const index_type& index) const;

        storage_type m_storage;
    };

    /***********************
     * xdefault_map_scheme *
     ***********************/

    template <class T, class I>
    struct xdefault_map_scheme
    {
        using index_type = I;
        using value_type = T;
        using size_type = typename index_type::value_type;
        using storage_type = std::map<index_type, value_type>;
        using type = xmap_scheme<storage_type>;
    };

    template <class T, class I>
    using xdefault_map_scheme_t = typename xdefault_map_scheme<T, I>::type;

    /***************************
     * xmap_scheme_nz_iterator *
     ***************************/

    namespace detail
    {
        template <class scheme>
        struct xmap_scheme_nz_iterator_types
        {
            using storage_type = typename scheme::storage_type;
            using index_type = typename scheme::index_type;
            using subiterator = typename storage_type::iterator;
            using value_type = typename storage_type::mapped_type;
            using reference = value_type&;
            using pointer = value_type*;
            using difference_type = typename storage_type::difference_type;
        };

        template <class scheme>
        struct xmap_scheme_nz_iterator_types<const scheme>
        {
            using storage_type = typename scheme::storage_type;
            using index_type = typename scheme::index_type;
            using subiterator = typename storage_type::const_iterator;
            using value_type = typename storage_type::mapped_type;
            using reference = const value_type&;
            using pointer = const value_type*;
            using difference_type = typename storage_type::difference_type;
        };
    }

    template <class scheme>
    class xmap_scheme_nz_iterator
        : public xtl::xrandom_access_iterator_base3<xmap_scheme_nz_iterator<scheme>,
                                                    detail::xmap_scheme_nz_iterator_types<scheme>>
    {
    public:

        using self_type = xmap_scheme_nz_iterator<scheme>;
        using scheme_type = scheme;
        using iterator_types = detail::xmap_scheme_nz_iterator_types<scheme>;
        using subiterator = typename iterator_types::subiterator;
        using index_type = typename iterator_types::index_type;
        using value_type = typename iterator_types::value_type;
        using reference = typename iterator_types::reference;
        using pointer = typename iterator_types::pointer;
        using difference_type = typename iterator_types::difference_type;

        xmap_scheme_nz_iterator();
        xmap_scheme_nz_iterator(scheme& s, subiterator it);

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

        static const value_type ZERO;

    private:

        scheme_type* p_scheme;
        subiterator m_it;
    };                                         

    template <class S>
    bool operator==(const xmap_scheme_nz_iterator<S>& lhs,
                    const xmap_scheme_nz_iterator<S>& rhs);

    template <class S>
    bool operator<(const xmap_scheme_nz_iterator<S>& lhs,
                   const xmap_scheme_nz_iterator<S>& rhs);

    /******************************
     * xmap_scheme implementation *
     ******************************/

    template <class ST>
    inline auto xmap_scheme<ST>::storage() const -> const storage_type&
    {
        return m_storage;
    }

    template <class ST>
    inline auto xmap_scheme<ST>::find_element(const index_type& index) -> pointer
    {
        return const_cast<pointer>(find_element_impl(index));
    }

    template <class ST>
    inline auto xmap_scheme<ST>::find_element(const index_type& index) const -> const_pointer
    {
        return find_element_impl(index);
    }

    template <class ST>
    inline void xmap_scheme<ST>::insert_element(const index_type& index, const_reference value)
    {
        m_storage.insert(std::make_pair(index, value));
    }

    template <class ST>
    inline void xmap_scheme<ST>::remove_element(const index_type& index)
    {
        m_storage.erase(m_storage.find(index));
    }

    template <class ST>
    template <class strides_type, class shape_type>
    inline void xmap_scheme<ST>::update_entries(const strides_type& old_strides,
                                                const strides_type& new_strides,
                                                const shape_type&)
    {
        storage_type new_storage;
        for (auto& old_entry: m_storage)
        {
            std::size_t offset = element_offset<std::size_t>(old_strides, old_entry.first.cbegin(), old_entry.first.cend());
            index_type new_index = unravel_from_strides(offset, new_strides);
            new_storage.insert(std::make_pair(new_index, old_entry.second));
        }
        using std::swap;
        swap(m_storage, new_storage);
    }

    template <class ST>
    inline auto xmap_scheme<ST>::nz_begin() -> nz_iterator
    {
        return nz_iterator(*this, m_storage.begin());
    }

    template <class ST>
    inline auto xmap_scheme<ST>::nz_end() -> nz_iterator
    {
        return nz_iterator(*this, m_storage.end());
    }

    template <class ST>
    inline auto xmap_scheme<ST>::nz_begin() const -> const_nz_iterator
    {
        return nz_cbegin();
    }

    template <class ST>
    inline auto xmap_scheme<ST>::nz_end() const -> const_nz_iterator
    {
        return nz_cend();
    }

    template <class ST>
    inline auto xmap_scheme<ST>::nz_cbegin() const -> const_nz_iterator
    {
        return const_nz_iterator(*this, m_storage.cbegin());
    }

    template <class ST>
    inline auto xmap_scheme<ST>::nz_cend() const -> const_nz_iterator
    {
        return const_nz_iterator(*this, m_storage.cend());
    }

    template <class ST>
    inline auto xmap_scheme<ST>::find_element_impl(const index_type& index) const -> const_pointer
    {
        auto it = m_storage.find(index);
        return it == m_storage.end() ? nullptr : &(it->second);
    }

    /******************************************
     * xmap_scheme_nz_iterator implementation *
     ******************************************/

    template <class scheme>
    const typename xmap_scheme_nz_iterator<scheme>::value_type
    xmap_scheme_nz_iterator<scheme>::ZERO = 0;

    template <class S>
    inline xmap_scheme_nz_iterator<S>::xmap_scheme_nz_iterator()
        : p_scheme(nullptr)
    {
    }

    template <class S>
    inline xmap_scheme_nz_iterator<S>::xmap_scheme_nz_iterator(S& s, subiterator it)
        : p_scheme(&s)
        , m_it(it)
    {
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::operator++() -> self_type&
    {
        ++m_it;
        return *this;
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::operator--() -> self_type&
    {
        --m_it;
        return *this;
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::operator+=(difference_type n) -> self_type&
    {
        std::advance(m_it, n);
        return *this;
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::operator-=(difference_type n) -> self_type&
    {
        std::advance(m_it, -n);
        return *this;
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::operator-(const self_type& rhs) const -> difference_type
    {
        return std::distance(m_it, rhs.m_it);
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::operator*() const -> reference
    {
        return m_it->second;
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::operator->() const -> pointer
    {
        return &(m_it->second);
    }

    template <class S>
    inline auto xmap_scheme_nz_iterator<S>::index() const -> const index_type&
    {
        return m_it->first;
    }

    template <class S>
    inline bool xmap_scheme_nz_iterator<S>::equal(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_it == rhs.m_it;
    }

    template <class S>
    inline bool xmap_scheme_nz_iterator<S>::less_than(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_it < rhs.m_it;
    }

    template <class S>
    inline bool operator==(const xmap_scheme_nz_iterator<S>& lhs,
                           const xmap_scheme_nz_iterator<S>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class S>
    inline bool operator<(const xmap_scheme_nz_iterator<S>& lhs,
                          const xmap_scheme_nz_iterator<S>& rhs)
    {
        return lhs.less_than(rhs);
    }
}

#endif


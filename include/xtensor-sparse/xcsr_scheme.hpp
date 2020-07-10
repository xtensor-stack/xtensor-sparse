#ifndef XSPARSE_CSR_SCHEME_HPP
#define XSPARSE_CSR_SCHEME_HPP

#include <type_traits>

#include <xtensor/xstorage.hpp>
#include <xtensor/xstrides.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace xt
{
    template <class scheme>
    class xcsr_scheme_nz_iterator;

    /***************************
     * xcsr_scheme declaration *
     ***************************/

    template <class P, class C, class ST>
    class xcsr_scheme
    {
    public:

        using self_type = xcsr_scheme<P, C, ST>;
        using position_type = P;
        using coordinate_type = C;
        using storage_type = ST;
        using index_type = std::array<std::size_t, 2>;

        using value_type = typename storage_type::value_type;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;

        using nz_iterator = xcsr_scheme_nz_iterator<self_type>;
        using const_nz_iterator = xcsr_scheme_nz_iterator<const self_type>;

        xcsr_scheme(std::size_t size);

        const position_type& position() const;
        const coordinate_type& coordinate() const;
        const storage_type& storage() const;

        storage_type& storage();

        pointer find_element(const index_type& index);
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

        position_type m_pos;
        coordinate_type m_coords;
        storage_type m_storage;

        friend class xcsr_scheme_nz_iterator<self_type>;
        friend class xcsr_scheme_nz_iterator<const self_type>;
    };

    /***************************************
     * xcsr_scheme_nz_iterator declaration *
     ***************************************/

    namespace detail
    {
        template <class scheme>
        struct xcsr_scheme_storage_type
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::iterator;
        };

        template <class scheme>
        struct xcsr_scheme_storage_type<const scheme>
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::const_iterator;
        };

        template <class scheme>
        struct xcsr_scheme_nz_iterator_types : xcsr_scheme_storage_type<scheme>
        {
            using base_type = xcsr_scheme_storage_type<scheme>;
            using index_type = typename scheme::index_type;
            using position_type = typename scheme::position_type;
            using position_iterator = typename position_type::const_iterator;
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
    class xcsr_scheme_nz_iterator: public xtl::xrandom_access_iterator_base3<xcsr_scheme_nz_iterator<scheme>,
                                                                          detail::xcsr_scheme_nz_iterator_types<scheme>>
    {
    public:
        using self_type = xcsr_scheme_nz_iterator;
        using xcsr_scheme = scheme;
        using iterator_types = detail::xcsr_scheme_nz_iterator_types<scheme>;
        using index_type = typename iterator_types::index_type;
        using position_type = typename iterator_types::position_type;
        using position_iterator = typename iterator_types::position_iterator;
        using coordinate_type = typename iterator_types::coordinate_type;
        using coordinate_iterator = typename iterator_types::coordinate_iterator;
        using value_type = typename iterator_types::value_type;
        using value_iterator = typename iterator_types::value_iterator;
        using reference = typename iterator_types::reference;
        using pointer = typename iterator_types::pointer;
        using difference_type = typename iterator_types::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        xcsr_scheme_nz_iterator(scheme& s, position_iterator&& pit, coordinate_iterator&& cit);

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

        index_type& update_current_index() const;

        position_iterator m_pit;
        coordinate_iterator m_cit;
        mutable index_type m_current_index;
        xcsr_scheme* p_scheme;
    };

    template <class scheme>
    bool operator==(const xcsr_scheme_nz_iterator<scheme>& lhs,
                    const xcsr_scheme_nz_iterator<scheme>& rhs);

    template <class scheme>
    bool operator<(const xcsr_scheme_nz_iterator<scheme>& lhs,
                   const xcsr_scheme_nz_iterator<scheme>& rhs);

    /******************************
     * xcsr_scheme implementation *
     ******************************/
    namespace detail
    {
        namespace csr
        {
            template <class Pos, class Coord, class Index>
            std::size_t insert_index(Pos& pos, Coord& coord, const Index& index)
            {        
                std::size_t ielem = index[0];
                auto it = std::find_if(coord.cbegin() + static_cast<std::ptrdiff_t>(pos[ielem]), coord.cbegin() + static_cast<std::ptrdiff_t>(pos[ielem + 1]), [&](auto e){return e >= index[1];});
                if (it == coord.cbegin() + static_cast<std::ptrdiff_t>(pos[ielem + 1]) || *it != index[1])
                {
                    for(std::size_t j = ielem + 1; j < pos.size(); ++j)
                    {
                        pos[j]++;
                    }
                    auto dst = static_cast<std::size_t>(std::distance(coord.cbegin(), it));
                    coord.insert(it, index[1]);
                    return dst;
                }
                return std::numeric_limits<std::size_t>::max();
            }
        }
    }

    template <class P, class C, class ST>
    inline xcsr_scheme<P, C, ST>::xcsr_scheme(std::size_t size): m_pos(size + 1, 0)
    {
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::position() const -> const position_type&
    {
        return m_pos;
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::coordinate() const -> const coordinate_type&
    {
        return m_coords;
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::storage() const -> const storage_type&
    {
        return m_storage;
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::find_element(const index_type& index) -> pointer
    {
        if (m_pos.size() == 0 || m_pos[index[0]] == m_pos[index[0]+1])
        {
            return nullptr;
        }

        std::size_t ielem = index[0];
        auto it = std::find(m_coords.cbegin() + static_cast<std::ptrdiff_t>(m_pos[ielem]), m_coords.cbegin() + static_cast<std::ptrdiff_t>(m_pos[ielem + 1]), index[1]);
        if (it != m_coords.cbegin() + static_cast<std::ptrdiff_t>(m_pos[ielem + 1]))
        {
            std::ptrdiff_t dst = std::distance(m_coords.cbegin(), it);
            return &(*(m_storage.begin() + dst));
        }

        return nullptr;
    }

    template <class P, class C, class ST>
    inline void xcsr_scheme<P, C, ST>::insert_element(const index_type& index, const_reference value)
    {
        XTENSOR_ASSERT(index.size() == 2);
        XTENSOR_ASSERT(m_pos.size() - 1 > index[0]);

        auto ielem = detail::csr::insert_index(m_pos, m_coords, index);
        XTENSOR_ASSERT(ielem != std::numeric_limits<std::size_t>::max());

        if (ielem == m_storage.size())
        {
            m_storage.push_back(value);
        }
        else
        {
            m_storage.insert(m_storage.cbegin() + static_cast<std::ptrdiff_t>(ielem), value);
        }

    }

    template <class P, class C, class ST>
    inline void xcsr_scheme<P, C, ST>::remove_element(const index_type& index)
    {
        // TODO: implement the version with a real remove inside coord, pos and storage
        auto elem = find_element(index);
        if (elem)
        {
            *elem = value_type(0);
        }
    }

    template <class P, class C, class ST>
    template <class strides_type, class shape_type>
    inline void xcsr_scheme<P, C, ST>::update_entries(const strides_type& old_strides,
                                                      const strides_type& new_strides,
                                                      const shape_type& new_shape)
    {
        coordinate_type new_coords;
        position_type new_pos(new_shape[0] + 1, 0);
        index_type old_index;

        for(std::size_t i = 0; i < m_pos.size() - 1; ++i)
        {
            old_index[0] = i;
            for(std::size_t j = m_pos[i]; j < m_pos[i+1]; ++j)
            {
                old_index[1] = m_coords[j];
                std::size_t offset = element_offset<std::size_t>(old_strides, old_index.cbegin(), old_index.cend());
                index_type new_index = xtl::forward_sequence<index_type, svector<std::size_t>>(unravel_from_strides(offset, new_strides));
                detail::csr::insert_index(new_pos, new_coords, new_index);
            }
        }

        using std::swap;
        swap(m_pos, new_pos);
        swap(m_coords, new_coords);
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::nz_begin() -> nz_iterator
    {
        return nz_iterator(*this, m_pos.cbegin(), m_coords.cbegin());
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::nz_end() -> nz_iterator
    {
        return nz_iterator(*this, m_pos.cend() - 2, m_coords.cend());
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::nz_begin() const -> const_nz_iterator
    {
        return nz_cbegin();
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::nz_end() const -> const_nz_iterator
    {
        return nz_cend();
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::nz_cbegin() const -> const_nz_iterator
    {
        return const_nz_iterator(*this, m_pos.cbegin(), m_coords.cbegin());
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::nz_cend() const -> const_nz_iterator
    {
        return const_nz_iterator(*this, m_pos.cend() - 2, m_coords.cend());
    }

    template <class P, class C, class ST>
    inline auto xcsr_scheme<P, C, ST>::storage() -> storage_type&
    {
        return m_storage;
    }

    /***************************************
     * xcsr_scheme_nz_iterator implementation *
     ***************************************/

    template <class scheme>
    inline xcsr_scheme_nz_iterator<scheme>::xcsr_scheme_nz_iterator(
        scheme& s, 
        position_iterator&& pit,
        coordinate_iterator&& cit)
        : m_pit(std::move(pit))
        , m_cit(std::move(cit))
        , p_scheme(&s)
    {
        update_current_index();
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::operator++() -> self_type&
    {
        ++m_cit;
        auto dst = static_cast<std::size_t>(std::distance(p_scheme->coordinate().cbegin(), m_cit));
        while (dst == *(m_pit + 1))
        {
            ++m_pit;
        }
        return *this;
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::operator--() -> self_type&
    {
        --m_cit;
        auto dst = static_cast<std::size_t>(std::distance(p_scheme->coordinate().cbegin(), m_cit));
        while (dst == *m_pit - 1)
        {
            --m_pit;
        }
        return *this;
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::operator+=(difference_type n) -> self_type&
    {
        m_cit += n;
        auto dst = static_cast<std::size_t>(std::distance(p_scheme->coordinate().cbegin(), m_cit));
        while (m_pit != p_scheme->position().cend() && dst >= *(m_pit + 1))
        {
            ++m_pit;
        }
        return *this;
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::operator-=(difference_type n) -> self_type&
    {
        m_cit -= n;
        auto dst = static_cast<std::size_t>(std::distance(p_scheme->coordinate().cbegin(), m_cit));
        while (m_pit != p_scheme->position().cbegin() && dst < *m_pit)
        {
            --m_pit;
        }
        return *this;
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_cit - rhs.m_cit;
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::operator*() const -> reference
    {
        std::ptrdiff_t dst = std::distance(p_scheme->coordinate().cbegin(), m_cit);
        return *(p_scheme->storage().begin() + dst);
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::operator->() const -> pointer
    {
        std::ptrdiff_t dst = std::distance(p_scheme->coordinate().cbegin(), m_cit);
        return &(p_scheme->storage().begin() + dst);
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::index() const -> const index_type&
    {
        return update_current_index();
    }

    template <class scheme>
    inline auto xcsr_scheme_nz_iterator<scheme>::update_current_index() const -> index_type&
    {
        m_current_index[0] = static_cast<std::size_t>(std::distance(p_scheme->position().cbegin(), m_pit));
        m_current_index[1] = *m_cit;
        return m_current_index;
    }

    template <class scheme>
    inline bool xcsr_scheme_nz_iterator<scheme>::equal(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_cit == rhs.m_cit;
    }

    template <class scheme>
    inline bool xcsr_scheme_nz_iterator<scheme>::less_than(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_cit < rhs.m_cit;
    }

    template <class scheme>
    inline bool operator==(const xcsr_scheme_nz_iterator<scheme>& lhs,
                           const xcsr_scheme_nz_iterator<scheme>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class scheme>
    inline bool operator<(const xcsr_scheme_nz_iterator<scheme>& lhs,
                          const xcsr_scheme_nz_iterator<scheme>& rhs)
    {
        return lhs.less_than(rhs);
    }
}
#endif

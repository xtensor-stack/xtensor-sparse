#ifndef XSPARSE_COO_SCHEME_HPP
#define XSPARSE_COO_SCHEME_HPP

#include <algorithm>

#include <xtensor/xstorage.hpp>
#include <xtensor/xstrides.hpp>

namespace xt
{
    /***************
     * xcoo_scheme *
     ***************/

    template <class P, class C, class ST, class IT = svector<std::size_t>>
    class xcoo_scheme
    {
    public:

        using position_type = P;
        using coordinate_type = C;
        using storage_type = ST;
        using index_type = IT;

        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;

        const position_type& position() const;
        const coordinate_type& coordinate() const;
        const storage_type& storage() const;

        xcoo_scheme();

        pointer find_element(const index_type& index);
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        template<class strides_type>
        void update_entries(const strides_type& old_strides,
                            const strides_type& new_strides);

    private:

        position_type m_pos;
        coordinate_type m_coords;
        storage_type m_storage;
    };

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
    template<class strides_type>
    inline void xcoo_scheme<P, C, ST, IT>::update_entries(const strides_type& old_strides,
                                                          const strides_type& new_strides)
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
}

#endif

#ifndef XSPARSE_MAP_SCHEME_HPP
#define XSPARSE_MAP_SCHEME_HPP

#include <xtensor/xstrides.hpp>

namespace xt
{
    /***************
     * xmap_scheme *
     ***************/

    template <class ST>
    class xmap_scheme
    {
    public:

        using storage_type = ST;
        using index_type = typename storage_type::key_type;
        using value_type = typename storage_type::mapped_type;
        using const_reference = const value_type&;
        using pointer = value_type*;

        const storage_type& storage() const;

        pointer find_element(const index_type& index);
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        template<class strides_type, class shape_type>
        void update_entries(const strides_type& old_strides,
                            const strides_type& new_strides,
                            const shape_type& new_shape);

    private:

        storage_type m_storage;
    };

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
        auto it = m_storage.find(index);
        return it == m_storage.end() ? nullptr : &(it->second);
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
    template<class strides_type, class shape_type>
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
}

#endif


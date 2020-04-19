#ifndef XSPARSE_COO_CONTAINER_HPP
#define XSPARSE_COO_CONTAINER_HPP

#include "xsparse_container.hpp"

namespace xt
{
    template <class D>
    class xcoo_container : public xsparse_container<D>
    {
    public:
        using base_type = xsparse_container<D>;
        using index_storage_type = typename base_type::index_storage_type;
        using storage_type = typename base_type::storage_type;
        using index_type = typename base_type::index_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;

        using true_pointer = value_type*;

        const index_storage_type& index_storage() const noexcept;
        index_storage_type& index_storage() noexcept;

        true_pointer find_element(const index_type& index);
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

    protected:

        using strides_type = typename base_type::strides_type;

        xcoo_container() noexcept;
        ~xcoo_container() = default;

        xcoo_container(const xcoo_container&) = default;
        xcoo_container& operator=(const xcoo_container&) = default;

        xcoo_container(xcoo_container&&) = default;
        xcoo_container& operator=(xcoo_container&&) = default;

        void update_entries(const strides_type& old_strides);
    private:

        index_storage_type m_index;

    };

    template <class D>
    inline auto xcoo_container<D>::index_storage() const noexcept -> const index_storage_type&
    {
        return m_index;
    }

    template <class D>
    inline auto xcoo_container<D>::index_storage() noexcept -> index_storage_type&
    {
        return m_index;
    }

    template <class D>
    inline void xcoo_container<D>::update_entries(const strides_type& old_strides)
    {
        // storage_type new_data;
        // for(auto& c: storage())
        // {
        //     auto& old_key = c.first;
        //     size_type index = element_offset<size_type>(old_strides, old_key.cbegin(), old_key.cend());
        //     shape_type new_key = unravel_from_strides(index, strides());
        //     new_data[new_key] = c.second;
        // }

        // std::swap(storage(), new_data);
    }

    template <class D>
    inline auto xcoo_container<D>::find_element(const index_type& index) -> true_pointer
    {
        auto it = std::find(m_index.begin(), m_index.end(), index);
        return it == m_index.end() ? nullptr : &(this->storage().begin() + (it - m_index));
    }

    template <class D>
    inline void xcoo_container<D>::insert_element(const index_type& index, const_reference value)
    {
        index_storage().push_back(index);
        this->storage().push_back(value);
    }

    template <class D>
    inline void xcoo_container<D>::remove_element(const index_type& index)
    {
        auto it = std::find(m_index.begin(), m_index.end(), index);
        if (it != m_index.end())
        {
            m_index.erase(it);
            this->storage().erase(this->storage().begin() + (it - m_index.begin()));
        }
    }

}
#endif
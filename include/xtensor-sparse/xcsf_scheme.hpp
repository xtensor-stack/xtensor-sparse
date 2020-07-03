#ifndef XSPARSE_CSF_SCHEME_HPP
#define XSPARSE_CSF_SCHEME_HPP


#include <xtensor/xstorage.hpp>
#include <xtensor/xstrides.hpp>

namespace xt
{
    template <class Scheme, bool is_const>
    class xcsf_scheme_iterator;

    /***************************
     * xcsf_scheme declaration *
     ***************************/
    template <class P, class C, class ST, class IT = svector<std::size_t>>
    class xcsf_scheme
    {
    public:

        using self_type = xcsf_scheme<P, C, ST, IT>;
        using position_type = P;
        using coordinate_type = C;
        using storage_type = ST;
        using index_type = IT;

        using value_type = typename storage_type::value_type;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;

        using iterator = xcsf_scheme_iterator<self_type, false>;
        using const_iterator = xcsf_scheme_iterator<self_type, true>;

        const position_type& position() const;
        const coordinate_type& coordinate() const;
        const storage_type& storage() const;

        pointer find_element(const index_type& index);
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        template<class strides_type>
        void update_entries(const strides_type& old_strides,
                            const strides_type& new_strides);

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
    };

    /************************************
     * xcsf_scheme_iterator declaration *
     ************************************/

    template <class Scheme, bool is_const>
    class xcsf_scheme_iterator
    {
    public:
        using self_type = xcsf_scheme_iterator;
        using xcsf_scheme = Scheme;
        using index_type = typename Scheme::index_type;
        using reference = std::conditional_t<is_const,
                                             typename Scheme::const_reference,
                                             typename Scheme::reference>;

        xcsf_scheme_iterator(Scheme& scheme, index_type&& pos_index, index_type&& coord_index);

        self_type& operator++();
        self_type& operator--();

        index_type& index() const;
        reference value() const;
        
    private:

        index_type& update_current_index() const;
        std::size_t position(std::size_t d, std::size_t i) const;
        std::size_t coordinate(std::size_t d, std::size_t i) const;

        index_type m_pos_index;
        index_type m_coord_index;
        mutable index_type m_current_index;
        xcsf_scheme* p_scheme;
    };

    /******************************
     * xcsf_scheme implementation *
     ******************************/

    namespace detail
    {
        template<class Pos, class Coord, class Func, class Index>
        void for_each_sparse_impl(std::integral_constant<std::size_t, 0>, std::size_t i, std::size_t ielem, const Pos& pos, const Coord& coord, Index& index, Func&& f)
        {
            for(std::size_t p = pos[i][ielem]; p<pos[i][ielem + 1]; ++p)
            {
                index[i] = coord[i][p];
                f(index);
            }
        }

        template<class Pos, class Coord, class Func, class Index>
        void for_each_sparse_impl(std::integral_constant<std::size_t, 1>, std::size_t i, std::size_t ielem, const Pos& pos, const Coord& coord, Index& index, Func&& f)
        {
            for(std::size_t p = pos[i][ielem]; p<pos[i][ielem + 1]; ++p)
            {
                index[i] = coord[i][p];
                if (i == pos.size() - 2)
                {
                    for_each_sparse_impl(std::integral_constant<std::size_t, 0>{}, i+1, p, pos, coord, index, std::forward<Func>(f));
                }
                else
                {
                    for_each_sparse_impl(std::integral_constant<std::size_t, 1>{}, i+1, p, pos, coord, index, std::forward<Func>(f));
                }
            }
        }

        template<class Pos, class Coord, class Func>
        void for_each(const Pos& pos, const Coord& coord, Func&& f)
        {
            std::vector<std::size_t> index(pos.size());
            if (pos.size() == 1)
            {
                for_each_sparse_impl(std::integral_constant<std::size_t, 0>{}, 0, 0, pos, coord, index, std::forward<Func>(f));
            }
            else if (pos.size() > 1)
            {
                for_each_sparse_impl(std::integral_constant<std::size_t, 1>{}, 0, 0, pos, coord, index, std::forward<Func>(f));
            }
        }

        template<class Pos, class Coord, class Index>
        std::size_t insert_index(Pos& pos, Coord& coord, const Index& index)
        {
            std::size_t ielem = 0;
            if (pos.size() == 0)
            {
                pos.resize(index.size());
                coord.resize(index.size());
                for(std::size_t i=0; i<index.size(); ++i)
                {
                    pos[i] = {0, 1};
                    coord[i] = {index[i]};
                }
            }
            else
            {
                for(std::size_t i=0; i<index.size(); ++i)
                {
                    auto it = std::find_if(coord[i].cbegin() + pos[i][ielem], coord[i].cbegin() + pos[i][ielem + 1], [&](auto e){return e >= index[i];});

                    if (it == coord[i].cbegin() + pos[i][ielem + 1] || *it != index[i])
                    {
                        for(std::size_t j=ielem + 1; j<pos[i].size(); ++j)
                        {
                            pos[i][j]++;
                        }
                        coord[i].insert(it, index[i]);

                        ielem = it - (coord[i].cbegin() + pos[i][ielem]);

                        for(std::size_t k=i+1; k<index.size(); ++k)
                        {
                            pos[k].insert(pos[k].begin() + ielem + 1, pos[k][ielem]);
                            for(std::size_t j=ielem + 1; j<pos[k].size(); ++j)
                            {
                                pos[k][j]++;
                            }   
                            coord[k].insert(coord[k].cbegin() + pos[k][ielem], index[k]);
                            ielem = pos[k][ielem];
                        }
                        return ielem;
                    }
                    ielem = it - (coord[i].cbegin() + pos[i][ielem]);
                }
                return std::numeric_limits<std::size_t>::max();
            }
            return ielem;
        }
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::position() const -> const position_type&
    {
        return m_pos;
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::coordinate() const -> const coordinate_type&
    {
        return m_coords;
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::storage() const -> const storage_type&
    {
        return m_storage;
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::find_element(const index_type& index) -> pointer
    {
        if (m_pos.size() == 0)
        {
            return nullptr;
        }

        std::size_t ielem = 0;
        for(std::size_t i=0; i<index.size(); ++i)
        {
            auto it = std::find(m_coords[i].cbegin() + m_pos[i][ielem], m_coords[i].cbegin() + m_pos[i][ielem + 1], index[i]);
            if (it != m_coords[i].cbegin() + m_pos[i][ielem + 1])
            {
                if (i == index.size() - 1)
                {
                    std::ptrdiff_t dst = std::distance(m_coords[i].cbegin(), it);
                    return &(*(m_storage.begin() + dst));
                }
                else
                {
                    ielem = it - (m_coords[i].cbegin() + m_pos[i][ielem]);
                }
            }
            else
            {
                return nullptr;
            }
        }
        return &m_storage[ielem];
    }

    template <class P, class C, class ST, class IT>
    inline void xcsf_scheme<P, C, ST, IT>::insert_element(const index_type& index, const_reference value)
    {
        if (m_pos.size() != 0)
        {
            XTENSOR_ASSERT(m_pos.size() == index.size());
        }

        auto ielem = detail::insert_index(m_pos, m_coords, index);
        XTENSOR_ASSERT(ielem != std::numeric_limits<std::size_t>::max());

        if (ielem == this->storage().size())
        {
            m_storage.push_back(value);
        }
        else
        {
            m_storage.insert(m_storage.cbegin() + static_cast<std::ptrdiff_t>(ielem), value);
        }
    }

    template <class P, class C, class ST, class IT>
    inline void xcsf_scheme<P, C, ST, IT>::remove_element(const index_type& index)
    {
        // TODO: implement the version with a real remove inside coord, pos and storage
        auto elem = find_element(index);
        if (elem)
        {
            *elem = value_type(0);
        }
    }

    template <class P, class C, class ST, class IT>
    template<class strides_type>
    inline void xcsf_scheme<P, C, ST, IT>::update_entries(const strides_type& old_strides,
                                                          const strides_type& new_strides)
    {
        coordinate_type new_coords;
        position_type new_pos;

        detail::for_each(m_pos, m_coords, [&](auto index){
            std::size_t offset = element_offset<std::size_t>(old_strides, index.cbegin(), index.cend());
            index_type new_index = unravel_from_strides(offset, new_strides);
            detail::insert_index(new_pos, new_coords, new_index);
        });

        using std::swap;
        swap(m_pos, new_pos);
        swap(m_coords, new_coords);
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::begin() -> iterator
    {
        index_type pos_index;
        index_type coord_index;
        
        return xcsf_scheme_iterator<self_type, false>(this, index_type(0), index_type(0));
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::end() const -> const_iterator
    {
        return cend();
    }
    
    /***************************************
     * xcsf_scheme_iterator implementation *
     ***************************************/

    template <class Scheme, bool is_const>
    inline xcsf_scheme_iterator<Scheme, is_const>::xcsf_scheme_iterator(Scheme& scheme, index_type&& pos_index, index_type&& coord_index)
        : p_scheme(&scheme)
        , m_pos_index(std::move(pos_index))
        , m_coord_index(std::move(coord_index))
    {
        update_current_index();
    }

    template <class Scheme, bool is_const>
    inline auto xcsf_scheme_iterator<Scheme, is_const>::operator++() -> self_type&
        {
            for (std::size_t i = m_coord_index.size(); i != std::size_t(0); --i)
            {
                std::size_t j = i - 1;
                if (m_coord_index[j] == p_scheme.coordinate().size()-1)
                {
                    *this = p_scheme->end();
                }
                ++m_coord_index[j];
                if (m_coord_index[j] == position(j, m_pos_index[j] + 1))
                {
                    ++m_pos_index[j];
                }
                else
                {
                    break;
                }
            }
            return *this;
        }

    template <class Scheme, bool is_const>
    inline auto xcsf_scheme_iterator<Scheme, is_const>::operator--() -> self_type&
    {
        for (std::size_t i = m_coord_index.size(); i != std::size_t(0); --i)
        {
            std::size_t j = i - 1;
            if (m_coord_index[j] == 0)
            {
                *this = p_scheme->rend();
            }
            --m_coord_index[j];
            if (m_coord_index[j] == position(j, m_pos_index[j]) - 1)
            {
                --m_pos_index[j];
            }
            else
            {
                break;
            }
        }
        return *this;
    }

    template <class Scheme, bool is_const>
    inline auto xcsf_scheme_iterator<Scheme, is_const>::index() const -> index_type&
    {
        return update_current_index();
    }

    template <class Scheme, bool is_const>
    inline auto xcsf_scheme_iterator<Scheme, is_const>::update_current_index() const -> index_type&
    {
        for(std::size_t d = 0; d < m_coord_index.size(); ++d)
        {
            m_current_index[d] = coordinate(d, m_coord_index[d]);
        }
        return m_current_index;
    }

    template <class Scheme, bool is_const>
    inline auto xcsf_scheme_iterator<Scheme, is_const>::value() const -> reference
    {
        return p_scheme->storage()[m_coord_index.back()];
    }

    template <class Scheme, bool is_const>
    inline std::size_t xcsf_scheme_iterator<Scheme, is_const>::position(std::size_t d, std::size_t i) const
    {
        return p_scheme->position()[d][i];
    }

    template <class Scheme, bool is_const>
    inline std::size_t xcsf_scheme_iterator<Scheme, is_const>::coordinate(std::size_t d, std::size_t i) const
    {
        return p_scheme->coordinate()[d][i];
    }
}
#endif
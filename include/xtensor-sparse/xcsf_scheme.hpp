#ifndef XSPARSE_CSF_SCHEME_HPP
#define XSPARSE_CSF_SCHEME_HPP

#include <iterator>
#include <type_traits>

#include <xtensor/xstorage.hpp>
#include <xtensor/xstrides.hpp>

namespace xt
{
    template <class scheme>
    class xcsf_scheme_nz_iterator;

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
        using const_pointer = typename storage_type::const_pointer;

        using nz_iterator = xcsf_scheme_nz_iterator<self_type>;
        using const_nz_iterator = xcsf_scheme_nz_iterator<const self_type>;

        const position_type& position() const;
        const coordinate_type& coordinate() const;
        const storage_type& storage() const;

        storage_type& storage();

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

        position_type m_pos;
        coordinate_type m_coords;
        storage_type m_storage;

        friend class xcsf_scheme_nz_iterator<self_type>;
        friend class xcsf_scheme_nz_iterator<const self_type>;
    };

    /***********************
     * xdefault_csf_scheme *
     ***********************/

    template <class T, class I>
    struct xdefault_csf_scheme
    {
        using index_type = I;
        using value_type = T;
        using size_type = typename index_type::value_type;
        using storage_type = std::vector<value_type>;
        using type = xcsf_scheme<std::vector<svector<size_type>>,
                                 std::vector<svector<size_type>>,
                                 storage_type,
                                 index_type>;
    };

    template <class T, class I>
    using xdefault_csf_scheme_t = typename xdefault_csf_scheme<T, I>::type;

    /***************************************
     * xcsf_scheme_nz_iterator declaration *
     ***************************************/

    namespace detail
    {
        template <class scheme>
        struct xcsf_scheme_storage_type
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::iterator;
        };

        template <class scheme>
        struct xcsf_scheme_storage_type<const scheme>
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::const_iterator;
        };

        template <class scheme>
        struct xcsf_scheme_nz_iterator_types : xcsf_scheme_storage_type<scheme>
        {
            using base_type = xcsf_scheme_storage_type<scheme>;
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
    class xcsf_scheme_nz_iterator: public xtl::xrandom_access_iterator_base3<xcsf_scheme_nz_iterator<scheme>,
                                                                             detail::xcsf_scheme_nz_iterator_types<scheme>>
    {
    public:

        using self_type = xcsf_scheme_nz_iterator;
        using xcsf_scheme = scheme;
        using iterator_types = detail::xcsf_scheme_nz_iterator_types<scheme>;
        using index_type = typename iterator_types::index_type;
        using index_type_iterator = svector<typename index_type::const_iterator>;
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

        xcsf_scheme_nz_iterator(scheme& s, index_type_iterator&& pos_index, index_type_iterator&& coord_index);

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

        index_type& update_current_index() const;

        index_type_iterator m_pos_index;
        index_type_iterator m_coord_index;
        mutable index_type m_current_index;
        xcsf_scheme* p_scheme;
    };

    template <class scheme>
    bool operator==(const xcsf_scheme_nz_iterator<scheme>& it1,
                    const xcsf_scheme_nz_iterator<scheme>& it2);

    template <class scheme>
    bool operator<(const xcsf_scheme_nz_iterator<scheme>& it1,
                   const xcsf_scheme_nz_iterator<scheme>& it2);

    /******************************
     * xcsf_scheme implementation *
     ******************************/

    namespace detail
    {
        template <class Pos, class Coord, class Func, class Index>
        void for_each_sparse_impl(std::integral_constant<std::size_t, 0>, std::size_t i, std::size_t ielem, const Pos& pos, const Coord& coord, Index& index, Func&& f)
        {
            for(std::size_t p = pos[i][ielem]; p<pos[i][ielem + 1]; ++p)
            {
                index[i] = coord[i][p];
                f(index);
            }
        }

        template <class Pos, class Coord, class Func, class Index>
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

        template <class Pos, class Coord, class Func>
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

        template <class Pos, class Coord, class Index>
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

                        ielem = static_cast<std::size_t>(it - (coord[i].cbegin() + pos[i][ielem]));

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
                    ielem = static_cast<std::size_t>(it - (coord[i].cbegin() + pos[i][ielem]));
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
        return const_cast<pointer>(find_element_impl(index));
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::find_element(const index_type& index) const -> const_pointer
    {
        return find_element_impl(index);
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
    template <class strides_type, class shape_type>
    inline void xcsf_scheme<P, C, ST, IT>::update_entries(const strides_type& old_strides,
                                                          const strides_type& new_strides,
                                                          const shape_type&)
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
    inline auto xcsf_scheme<P, C, ST, IT>::find_element_impl(const index_type& index) const -> const_pointer
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
                    ielem = static_cast<std::size_t>(it - (m_coords[i].cbegin() + m_pos[i][ielem]));
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
    inline auto xcsf_scheme<P, C, ST, IT>::nz_begin() -> nz_iterator
    {
        std::size_t dim =  m_pos.size();
        typename nz_iterator::index_type_iterator pos_index(dim);
        typename nz_iterator::index_type_iterator coord_index(dim);
        for(std::size_t d = 0; d < dim; ++d)
        {
            pos_index[d] = m_pos[d].cbegin();
            coord_index[d] = m_coords[d].cbegin();
        }
        return nz_iterator(*this, std::move(pos_index), std::move(coord_index));
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::nz_end() -> nz_iterator
    {
        std::size_t dim =  m_pos.size();
        typename nz_iterator::index_type_iterator pos_index(dim);
        typename nz_iterator::index_type_iterator coord_index(dim);
        for(std::size_t d = 0; d < dim; ++d)
        {
            pos_index[d] = m_pos[d].cend() - 2;
            coord_index[d] = m_coords[d].cend() - 1;
        }
        ++coord_index.back();

        return nz_iterator(*this, std::move(pos_index), std::move(coord_index));
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::nz_begin() const -> const_nz_iterator
    {
        return nz_cbegin();
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::nz_end() const -> const_nz_iterator
    {
        return nz_cend();
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::nz_cbegin() const -> const_nz_iterator
    {
        std::size_t dim =  m_pos.size();
        typename const_nz_iterator::index_type_iterator pos_index(dim);
        typename const_nz_iterator::index_type_iterator coord_index(dim);
        for(std::size_t d = 0; d < dim; ++d)
        {
            pos_index[d] = m_pos[d].cbegin();
            coord_index[d] = m_coords[d].cbegin();
        }
        return const_nz_iterator(*this, std::move(pos_index), std::move(coord_index));
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::nz_cend() const -> const_nz_iterator
    {
        std::size_t dim =  m_pos.size();
        typename const_nz_iterator::index_type_iterator pos_index(dim);
        typename const_nz_iterator::index_type_iterator coord_index(dim);
        for(std::size_t d = 0; d < dim; ++d)
        {
            pos_index[d] = m_pos[d].cend() - 2;
            coord_index[d] = m_coords[d].cend() - 1;
        }
        ++coord_index.back();

        return const_nz_iterator(*this, std::move(pos_index), std::move(coord_index));
    }

    template <class P, class C, class ST, class IT>
    inline auto xcsf_scheme<P, C, ST, IT>::storage() -> storage_type&
    {
        return m_storage;
    }

    /******************************************
     * xcsf_scheme_nz_iterator implementation *
     ******************************************/

    template <class scheme>
    const typename xcsf_scheme_nz_iterator<scheme>::value_type
    xcsf_scheme_nz_iterator<scheme>::ZERO = 0;

    template <class scheme>
    inline xcsf_scheme_nz_iterator<scheme>::xcsf_scheme_nz_iterator(
        scheme& s, 
        index_type_iterator&& pos_index,
        index_type_iterator&& coord_index)
        : m_pos_index(std::move(pos_index))
        , m_coord_index(std::move(coord_index))
        , p_scheme(&s)
    {
        m_current_index.resize(m_pos_index.size());
        update_current_index();
    }

    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::operator++() -> self_type&
    {
        for (std::size_t i = m_coord_index.size(); i != std::size_t(0); --i)
        {
            std::size_t d = i - 1;
            if (m_coord_index[d] == p_scheme->coordinate()[d].end())
            {
                break;
            }
            ++m_coord_index[d];

            auto dst = static_cast<std::size_t>(std::distance(p_scheme->coordinate()[d].begin(), m_coord_index[d]));
            if (dst == *(m_pos_index[d] + 1))
            {
                ++m_pos_index[d];
            }
            else
            {
                break;
            }
        }
        return *this;
    }

    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::operator--() -> self_type&
    {
        for (std::size_t i = m_coord_index.size(); i != std::size_t(0); --i)
        {
            std::size_t d = i - 1;
            if (m_coord_index[d] == p_scheme->coordinate()[d].begin())
            {
                --m_coord_index[d];
                break;
            }
            --m_coord_index[d];

            auto dst = static_cast<std::size_t>(std::distance((p_scheme->coordinate()[d]).begin(), m_coord_index[d]));
            if (dst == *m_pos_index[d] - 1)
            {
                --m_pos_index[d];
            }
            else
            {
                break;
            }
        }
        return *this;
    }
    
    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::operator+=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            ++(*this);
        }
        return *this;
    }
    
    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::operator-=(difference_type n) -> self_type&
    {
        for (difference_type i = 0; i < n; ++i)
        {
            --(*this);
        }
        return *this;
    }

    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_coord_index.back() - rhs.m_coord_index.back();
    }

    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::operator*() const -> reference
    {
        std::ptrdiff_t dst = std::distance(p_scheme->coordinate().back().begin(), m_coord_index.back());
        return *(p_scheme->storage().begin() + dst);
    }

    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::operator->() const -> pointer
    {
        return &(this->operator*());
    }

    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::index() const -> const index_type&
    {
        return update_current_index();
    }

    template <class scheme>
    inline bool xcsf_scheme_nz_iterator<scheme>::equal(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_coord_index.back() == rhs.m_coord_index.back();
    }

    template <class scheme>
    inline bool xcsf_scheme_nz_iterator<scheme>::less_than(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_coord_index.back() < rhs.m_coord_index.back();
    }
    
    template <class scheme>
    inline auto xcsf_scheme_nz_iterator<scheme>::update_current_index() const -> index_type&
    {
        for(std::size_t d = 0; d < m_coord_index.size(); ++d)
        {
            m_current_index[d] = *m_coord_index[d];
        }
        return m_current_index;
    }

    template <class scheme>
    inline bool operator==(const xcsf_scheme_nz_iterator<scheme>& it1,
                           const xcsf_scheme_nz_iterator<scheme>& it2)
    {
        return it1.equal(it2);
    }

    template <class scheme>
    inline bool operator<(const xcsf_scheme_nz_iterator<scheme>& it1,
                          const xcsf_scheme_nz_iterator<scheme>& it2)
    {
        return it1.less_than(it2);
    }    
}

#endif

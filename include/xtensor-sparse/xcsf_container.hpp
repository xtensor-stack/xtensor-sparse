#ifndef XSPARSE_CSF_CONTAINER_HPP
#define XSPARSE_CSF_CONTAINER_HPP

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

#include "xsparse_container.hpp"

namespace xt
{
    /******************************
     * xcsf_container declaration *
     ******************************/

    template <class D>
    class xcsf_container : public xsparse_container<D>
    {
    public:
        using base_type = xsparse_container<D>;
        using index_storage_type = typename base_type::index_storage_type;
        using position_type = typename base_type::inner_types::position_type;
        using storage_type = typename base_type::storage_type;
        using index_type = typename base_type::index_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using size_type = typename base_type::size_type;

        using true_pointer = value_type*;

        const index_storage_type& index_storage() const noexcept;
        index_storage_type& index_storage() noexcept;


    protected:

        using strides_type = typename base_type::strides_type;

        xcsf_container() = default;
        ~xcsf_container() = default;

        xcsf_container(const xcsf_container&) = default;
        xcsf_container& operator=(const xcsf_container&) = default;

        xcsf_container(xcsf_container&&) = default;
        xcsf_container& operator=(xcsf_container&&) = default;

        void update_entries(const strides_type& old_strides);

        template<class... Args>
        const_reference access_impl(Args... args) const;
        template<class... Args>
        reference access_impl(Args... args);

    private:

        static const value_type ZERO;

        position_type m_pos;
        index_storage_type m_coord;

        true_pointer find_element(const index_type& index);
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        friend class xsparse_reference<xcsf_container<D>>;
    };

    /*********************************
     * xcsf_container implementation *
     *********************************/

    template<class D>
    const typename xcsf_container<D>::value_type xcsf_container<D>::ZERO = 0;

    template <class D>
    inline auto xcsf_container<D>::index_storage() const noexcept -> const index_storage_type&
    {
        return m_coord;
    }

    template <class D>
    inline auto xcsf_container<D>::index_storage() noexcept -> index_storage_type&
    {
        return m_coord;
    }

    namespace detail
    {
        template<std::size_t N, class Pos, class Coord, class Func, class Index>
        void for_each_sparse_impl(std::integral_constant<std::size_t, N>, std::size_t, std::size_t, const Pos&, const Coord&, Index& index, Func&&){}

        template<class Pos, class Coord, class Func, class Index>
        void for_each_sparse_impl(std::integral_constant<std::size_t, 1>, std::size_t i, std::size_t ielem, const Pos& pos, const Coord& coord, Index& index, Func&& f)
        {
            for(std::size_t p = pos[i][ielem]; p<pos[i][ielem + 1]; ++p)
            {
                index[i] = coord[i][p];
                if (i+1 == pos.size() - 1)
                {
                    for_each_sparse_impl(std::integral_constant<std::size_t, 1>{}, i+1, p, pos, coord, index, std::forward<Func>(f));
                }
                else
                {
                    for_each_sparse_impl(std::integral_constant<std::size_t, 0>{}, i+1, p, pos, coord, index, std::forward<Func>(f));
                }
                
            }
        }

        template<class Pos, class Coord, class Func, class Index>
        void for_each_sparse_impl(std::integral_constant<std::size_t, 0>, std::size_t i, std::size_t ielem, const Pos& pos, const Coord& coord, Index& index, Func&& f)
        {
            for(std::size_t p = pos[i][ielem]; p<pos[i][ielem + 1]; ++p)
            {
                index[i] = coord[i][p];
                f(index);
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

    template <class D>
    inline void xcsf_container<D>::update_entries(const strides_type& old_strides)
    {
        index_storage_type new_coord;
        position_type new_pos;
        
        detail::for_each(m_pos, m_coord, [&](auto index){
            size_type offset = element_offset<size_type>(old_strides, index.cbegin(), index.cend());
            shape_type new_index = unravel_from_strides(offset, this->strides());
            detail::insert_index(new_pos, new_coord, new_index);
        });

        using std::swap;
        swap(m_pos, new_pos);
        swap(m_coord, new_coord);
    }

    template <class D>
    inline auto xcsf_container<D>::find_element(const index_type& index) -> true_pointer
    {
        if (m_pos.size() == 0)
        {
            return nullptr;
        }

        std::size_t ielem = 0;
        for(std::size_t i=0; i<index.size(); ++i)
        {
            auto it = std::find(m_coord[i].cbegin() + m_pos[i][ielem], m_coord[i].cbegin() + m_pos[i][ielem + 1], index[i]);
            if (it != m_coord[i].cbegin() + m_pos[i][ielem + 1])
            {
                if (i == index.size() - 1)
                {
                    std::ptrdiff_t dst = std::distance(m_coord[i].cbegin(), it);
                    return &(*(this->storage().begin() + dst));
                }
                else
                {
                    ielem = it - (m_coord[i].cbegin() + m_pos[i][ielem]);
                }
            }
            else
            {
                return nullptr;
            }
        }
        return &this->storage()[ielem];
    }

    template <class D>
    inline void xcsf_container<D>::insert_element(const index_type& index, const_reference value)
    {
        auto ielem = detail::insert_index(m_pos, m_coord, index);
        if (ielem == std::numeric_limits<std::size_t>::max())
        {
            throw std::runtime_error("This should not happen");
        }
        if (ielem == this->storage().size())
        {
            this->storage().push_back(value);
        }
        else
        {
            this->storage().insert(this->storage().cbegin() + static_cast<std::ptrdiff_t>(ielem), value);
        }
    }

    template <class D>
    inline void xcsf_container<D>::remove_element(const index_type& index)
    {
        /*std::size_t ielem = 0;
        for(std::size_t i=0; i<index.size(); ++i)
        {
            auto it = std::find(m_coord[i].cbegin() + m_pos[i][ielem], m_coord[i].cbegin() + m_pos[i][ielem + 1], index[i]);
            if (it != m_coord[i].cbegin() + m_pos[i][ielem + 1])
            {
                if (i == index.size() - 1)
                {
                    std::ptrdiff_t dst = std::distance(m_coord[i].cbegin(), it);
                    this->storage().erase(this->storage().begin() + dst);
                }
                else
                {
                    ielem = it - (m_coord[i].cbegin() + m_pos[i][ielem]);
                }
            }
        }*/
        // TODO: implement the version with a real remove inside coord, pos and storage
        (*find_element(index)) = value_type(0);
    }

    template<class D>
    template<class... Args>
    inline auto xcsf_container<D>::access_impl(Args... args) const -> const_reference
    {
        // TODO: check if all args have the good type
        index_type key{static_cast<size_type>(args)...};

        auto it = find_element(key);
        if (it)
        {
            return *it;
        }
        return ZERO;
    }

    template<class D>
    template<class... Args>
    inline auto xcsf_container<D>::access_impl(Args... args) -> reference
    {
        // TODO: check if all args have the good type
        index_type key{static_cast<size_type>(args)...};

        auto it = find_element(key);
        value_type v = (it)? *it: value_type();
        return reference(*this, std::move(key), v);
    }
}

#endif
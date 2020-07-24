#ifndef XSPARSE_SCALAR_HPP
#define XSPARSE_SCALAR_HPP

#include <xtensor/xscalar.hpp>

namespace xt
{
    /**********************************
     * xscalar_nz_iterator definition *
     **********************************/

    template <bool is_const, class CT>
    class xscalar_nz_iterator
    {
    public:

        using self_type = xscalar_nz_iterator<is_const, CT>;
        using parent_type = xdummy_iterator<is_const, CT>;
        using storage_type = typename parent_type::storage_type;

        using value_type = typename storage_type::value_type;
        using reference = typename parent_type::reference;
        using pointer = typename parent_type::pointer;
        using difference_type = typename storage_type::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        using index_type = svector<std::size_t>;

        xscalar_nz_iterator(storage_type* c) noexcept;

        index_type index() const;

        self_type& operator++() noexcept;
        self_type& operator--() noexcept;

        reference operator*() const noexcept;

        bool equal(const self_type& rhs) const noexcept;
        bool less_than(const self_type& rhs) const noexcept;

    private:

        storage_type* p_c;
    };

    template <bool is_const, class CT>
    bool operator==(const xscalar_nz_iterator<is_const, CT>& lhs,
                    const xscalar_nz_iterator<is_const, CT>& rhs) noexcept;

    template <bool is_const, class CT>
    bool operator<(const xscalar_nz_iterator<is_const, CT>& lhs,
                   const xscalar_nz_iterator<is_const, CT>& rhs) noexcept;

    /**************************************
     * xscalar_nz_iterator implementation *
     **************************************/

    template <bool is_const, class CT>
    xscalar_nz_iterator<is_const, CT>::xscalar_nz_iterator(storage_type* c) noexcept
    : p_c{c}
    {}

    template <bool is_const, class CT>
    inline auto xscalar_nz_iterator<is_const, CT>::index() const -> index_type
    {
        return {};
    }

    template <bool is_const, class CT>
    inline auto xscalar_nz_iterator<is_const, CT>::operator++() noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xscalar_nz_iterator<is_const, CT>::operator--() noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xscalar_nz_iterator<is_const, CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <bool is_const, class CT>
    inline bool xscalar_nz_iterator<is_const, CT>::equal(const self_type& rhs) const noexcept
    {
        return p_c == rhs.p_c;
    }

    template <bool is_const, class CT>
    inline bool xscalar_nz_iterator<is_const, CT>::less_than(const self_type& rhs) const noexcept
    {
        return p_c < rhs.p_c;
    }

    template <bool is_const, class CT>
    inline bool operator==(const xscalar_nz_iterator<is_const, CT>& lhs,
                           const xscalar_nz_iterator<is_const, CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class CT>
    inline bool operator<(const xscalar_nz_iterator<is_const, CT>& lhs,
                          const xscalar_nz_iterator<is_const, CT>& rhs) noexcept
    {
        return lhs.less_than(rhs);
    }

    /*******************************
     * get_nz__begin / get_nz__end *
     *******************************/

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto get_nz_begin(xscalar<CT>& c) noexcept
    {
        return xscalar_nz_iterator<false, CT>(&c);
    }

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto get_nz_end(xscalar<CT>& c) noexcept
    {
        return xscalar_nz_iterator<false, CT>(&c);
    }

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto get_nz_begin(const xscalar<CT>& c) noexcept
    {
        return xscalar_nz_iterator<true, CT>(&c);
    }

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto get_nz_end(const xscalar<CT>& c) noexcept
    {
        return xscalar_nz_iterator<true, CT>(&c);
    }
}
#endif
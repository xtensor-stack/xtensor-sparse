#ifndef XSPARSE_FUNCTION_HPP
#define XSPARSE_FUNCTION_HPP

#include <algorithm>

#include <xtl/xmeta_utils.hpp>
#include <xtensor/xoperation.hpp>

#include "xsparse_expression.hpp"
#include "xutils.hpp"

#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

namespace xt
{

    /************************************
    * xfunction_nz_iterator declaration *
    *************************************/
    template <class F, class... CT>
    class xfunction_nz_iterator: public xtl::xrandom_access_iterator_base<xfunction_nz_iterator<F, CT...>,
                                                                          typename xfunction<F, CT...>::value_type,
                                                                          typename xfunction<F, CT...>::difference_type,
                                                                          typename xfunction<F, CT...>::pointer,
                                                                          typename xfunction<F, CT...>::reference>
    {
    public:

        using self_type = xfunction_nz_iterator<F, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction<F, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::random_access_iterator_tag;
        // TODO xtensor container doesn't have index_type
        // so we take shape_type which has the same dimension of index_type
        // and it's common. Check if specialization is needed...
        using index_type = promote_shape_t<typename std::decay_t<CT>::shape_type...>;

        template <class... It>
        xfunction_nz_iterator(const xfunction_type* func, It&&... it);

        self_type& operator++();
        // self_type& operator--();

        reference operator*() const;
        const index_type& index() const;

    private:

        static const value_type ZERO;

        template <class I>
        typename I::reference get_value(I* it) const;

        template <std::size_t... Is>
        reference apply(std::index_sequence<Is...>) const;

        const xfunction_type* p_f;
        index_type current_index;
        std::tuple<typename std::decay_t<CT>::const_nz_iterator...> m_nz_iterators;
        // Contains a pointer to the iterator in m_nz_iterators or nullptr
        std::tuple<typename std::decay_t<CT>::const_nz_iterator*...> m_nz_current_iterators;
    };

    namespace extension
    {
        namespace detail
        {
            namespace mpl = xtl::mpl;
            
            // Because of shitty VS2015 bug:
            // pack expansions cannot be used as arguments to non-packed parameters in alias templates
            template <class... CT>
            struct assign_tag_list
            {
                using type = mpl::vector<get_assign_tag_t<CT>...>;
            };

            template <class... CT>
            using assign_tag_list_t = typename assign_tag_list<CT...>::type;

            template <class... CT>
            struct get_default_assign_tag
            {
                using tag_list = assign_tag_list_t<CT...>;
                using type = std::conditional_t
                             <
                                    mpl::contains<tag_list, xdense_assign_tag>::value,
                                    xdense_assign_tag,
                                    xsparse_assign_tag
                             >;
            };

            template <class CT>
            struct get_default_assign_tag<CT>
                : get_assign_tag<CT>
            {
            };

            template <class... CT>
            struct get_multiply_assign_tag
            {
                using tag_list = assign_tag_list_t<CT...>;
                using type = std::conditional_t<
                                    mpl::contains<tag_list, xsparse_assign_tag>::value,
                                    xsparse_assign_tag,
                                    xdense_assign_tag>;
            };

            template <class F, class... CT>
            struct get_function_assign_tag : get_default_assign_tag<CT...>
            {
            };

            template <class... CT>
            struct get_function_assign_tag<xt::detail::multiplies, CT...>
                : get_multiply_assign_tag<CT...>
            {
            };

            template <class... CT>
            struct get_function_assign_tag<xt::detail::divides, CT...>
                : get_multiply_assign_tag<CT...>
            {
            };

            template <class F, class... CT>
            using get_function_assign_tag_t = typename get_function_assign_tag<F, CT...>::type;
        }

        /*************************
        *  xfunction_sparse_base *
        **************************/

        template<class F, class... CT>
        struct xfunction_sparse_base : public xsparse_empty_base<xfunction<F, CT...>>
        {
            using assign_tag = detail::get_function_assign_tag_t<F, CT...>;

            using const_nz_iterator = xfunction_nz_iterator<F, CT...>;
            using nz_iterator = const_nz_iterator;

            const_nz_iterator nz_begin() const;
            const_nz_iterator nz_cbegin() const;

        private:

            template<std::size_t... I>
            const_nz_iterator build_nz_iterator(std::index_sequence<I...>) const noexcept;

        };

    }

    namespace detail
    {
        template <class F, class... E>
        struct select_xfunction_expression<::xt::xsparse_expression_tag, F, E...>
        {
            using type = xfunction<F, E...>;
        };
    }

    /***************************************
    * xfunction_nz_iterator implementation *
    ****************************************/

    template <class F, class... CT>
    const typename xfunction_nz_iterator<F, CT...>::value_type
    xfunction_nz_iterator<F, CT...>::ZERO = 0;

    template <class F, class... CT>
    template <class... It>
    inline xfunction_nz_iterator<F, CT...>::xfunction_nz_iterator(const xfunction_type* func, It&&... it)
        : p_f(func),
          m_nz_iterators(std::forward<It>(it)...),
          m_nz_current_iterators(std::make_tuple(static_cast<typename std::decay_t<CT>::const_nz_iterator*>(nullptr)...))
    {
        auto min = [](const auto& init, const auto& iter)
        {
            return std::lexicographical_compare(init.cbegin(), init.cend(), iter.index().cbegin(), iter.index().cend()) ?
                init :
                iter.index();
        };
        current_index = xt::accumulate(min, index_type(p_f->dimension(), std::size_t(-1)), m_nz_iterators);

        auto ft = [this](auto& it){return (it.index() == current_index)? &it: nullptr;};
        transform(ft, m_nz_iterators, m_nz_current_iterators);
    }

    template <class F, class... CT>
    inline auto xfunction_nz_iterator<F, CT...>::operator++() -> self_type&
    {
        auto ft1 = [this](auto& it){if (it.index() == current_index) ++it; return nullptr;};
        transform(ft1, m_nz_iterators, m_nz_current_iterators);

        auto min = [](const auto& init, const auto& iter)
        {
            return std::lexicographical_compare(init.cbegin(), init.cend(), iter.index().cbegin(), iter.index().cend()) ?
                init :
                iter.index();
        };
        current_index = xt::accumulate(min, index_type(p_f->dimension(), std::size_t(-1)), m_nz_iterators);

        auto ft2 = [this](auto& it){return (it.index() == current_index)? &it: nullptr;};
        transform(ft2, m_nz_iterators, m_nz_current_iterators);
        return *this;
    }

    template <class F, class... CT>
    inline auto xfunction_nz_iterator<F, CT...>::operator*() const -> reference
    {
        return apply(std::make_index_sequence<sizeof...(CT)>{});
    }

    template <class F, class... CT>
    inline auto xfunction_nz_iterator<F, CT...>::index() const -> const index_type&
    {
        return current_index;
    }

    template <class F, class... CT>
    template <class I>
    inline auto xfunction_nz_iterator<F, CT...>::get_value(I* it) const -> typename I::reference
    {
        return (it == nullptr) ? ZERO : *(*it);
    }

    template <class F, class... CT>
    template <std::size_t... Is>
    inline auto xfunction_nz_iterator<F, CT...>::apply(std::index_sequence<Is...>) const -> reference
    {
        return (p_f->functor())(get_value(std::get<Is>(m_nz_current_iterators))...);
    }

    /****************************************
    *  xfunction_sparse_base implementation *
    *****************************************/

    namespace extension
    {
        template<class F, class... CT>
        inline auto xfunction_sparse_base<F, CT...>::nz_begin() const -> const_nz_iterator
        {
            return nz_cbegin();
        }

        template<class F, class... CT>
        inline auto xfunction_sparse_base<F, CT...>::nz_cbegin() const -> const_nz_iterator
        {
            return build_nz_iterator(std::make_index_sequence<sizeof...(CT)>());
        }

        template<class F, class... CT>
        template<std::size_t... I>
        inline auto xfunction_sparse_base<F, CT...>::build_nz_iterator(std::index_sequence<I...>) const noexcept -> const_nz_iterator
        {
            auto& args = this->derived_cast().arguments();
            return const_nz_iterator(&(this->derived_cast()), std::get<I>(args).nz_cbegin()...);
        }

        template <class F, class... CT>
        struct xfunction_base_impl<xsparse_expression_tag, F, CT...>
        {
            using type = xfunction_sparse_base<F, CT...>;
        };
    }
}

#endif
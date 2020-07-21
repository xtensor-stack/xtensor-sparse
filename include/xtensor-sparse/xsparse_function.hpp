#ifndef XSPARSE_FUNCTION_HPP
#define XSPARSE_FUNCTION_HPP

#include <xtl/xmeta_utils.hpp>
#include <xtensor/xoperation.hpp>

#include "xsparse_expression.hpp"

namespace xt
{
    template <class F, class... CT>
    class xfunction_nz_iterator
    {
    public:

        using self_type = xfunction_nz_iterator<F, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction<F, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using difference_type = typename xfunction_type::difference_type;
        using index_type = promote_shape_t<typename std::decay_t<CT>::index_type...>;

        template <class... It>
        xfunction_nz_iterator(const xfunction_type* func, It&&... it)
        : p_f(func), m_nz_iterators(std::forward<It>(it)...), m_nz_current_iterators(std::decay_t<CT>::const_nz_iterator*(nullptr)...)
        {}

        // self_type& operator++();
        // self_type& operator--();

        
        self_type& operator++()
        {
            auto f = [](auto& it){ if (it != nullptr){ ++(*it); it = nullptr;}};
            for_each(f, m_nz_current_iterators);

            auto min = [](const auto& init, const auto& iter)
            {
                return lexicographical_compare(init.cbegin(), init.cend(), iter.index().cbegin(), iter.index.cend()) ?
                  init :
                  decltype(init)(iter.index().cbegin(), iter.index().cend());
            };
            index_type current_index = xt::accumulate(min, index_type(p_f->dimension(), std::size_t(-1)), m_nz_iterators);

            auto ft = [&current_index](auto& it){return (it.index() == current_index)? it: nullptr;};
            transform(ft, m_nz_iterators, m_nz_current_iterators);
        }

        reference operator*()
        {
            return apply(std::make_index_sequence<sizeof...(CT)>{});
        }
    private:

        template <class I>
        typename I::reference get_value(I* it)
        {
            return it == nullptr ? value_type(0) : *(*it);
        }

        template <std::size_t... Is>
        auto apply(std::index_sequence<Is...>)
        {
            return *(p_f(get_value(std::get<Is>(m_nz_current_iterators))...));
        }

        xfunction_type* p_f;
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

        template<class F, class... CT>
        struct xfunction_sparse_base : public xsparse_empty_base<xfunction<F, CT...>>
        {
            using assign_tag = detail::get_function_assign_tag_t<F, CT...>;

            using const_nz_storage_iterator = xfunction_nz_iterator<F, CT...>;
            using nz_storage_iterator = const_nz_storage_iterator;

            const_nz_storage_iterator nz_begin()
            {
                auto f = [](const auto& e) noexcept { return e.nz_begin(); };
                return this->derived_cast().build_iterator(f, std::make_index_sequence<sizeof...(CT)>());
            }

            // friend class xsparse_empty_base<xfunction<F, CT...>>;
        };

        template <class F, class... CT>
        struct xfunction_base_impl<xsparse_expression_tag, F, CT...>
        {
            using type = xfunction_sparse_base<F, CT...>;
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
}

#endif
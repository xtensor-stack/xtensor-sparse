#ifndef XSPARSE_EXPRESSION_HPP
#define XSPARSE_EXPRESSION_HPP

#include <xtensor/xexpression.hpp>

namespace xt
{
    /**************************
     * xsparse_expression_tag *
     **************************/

    struct xsparse_expression_tag
    {
    };

    template<class E>
    struct is_xsparse_expression: std::is_same<xexpression_tag_t<E>, xsparse_expression_tag>
    {
    };

    template<class... E>
    struct xsparse_comparable : xtl::disjunction<is_xsparse_expression<E>...>
    {
    };

    namespace extension
    {
        /**********************
         * xsparse_assign_tag *
         **********************/

        struct xsparse_assign_tag
        {
        };

        struct xdense_assign_tag
        {
        };

        template <class E, class = void_t<int>>
        struct get_assign_tag
        {
            using type = xdense_assign_tag;
        };

        template <class E>
        struct get_assign_tag<E, xt::void_t<typename std::decay_t<E>::assign_tag>>
        {
            using type = typename std::decay_t<E>::assign_tag;
        };

        template <class E>
        using get_assign_tag_t = typename get_assign_tag<E>::type;

        /*********************************
         * xsparse_empty_base definition *
         *********************************/

        template <class D>
        class xsparse_empty_base
        {
        public:

            using expression_tag = xsparse_expression_tag;

        protected:

            D& derived_cast() noexcept;
            const D& derived_cast() const noexcept;
        };

        /*************************************
         * xsparse_empty_base implementation *
         *************************************/

        template <class D>
        inline D& xsparse_empty_base<D>::derived_cast() noexcept
        {
            return *static_cast<D*>(this);
        }

        template <class D>
        inline const D& xsparse_empty_base<D>::derived_cast() const noexcept
        {
            return *static_cast<const D*>(this);
        }
    }

}

#endif
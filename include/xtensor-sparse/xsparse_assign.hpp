#ifndef XSPARSE_ASSIGN_HPP
#define XSPARSE_ASSIGN_HPP

#include <xtensor/xassign.hpp>

#include "xsparse_expression.hpp"

namespace xt
{
    template <class T1, class T2>
    struct xsparse_assigner: public xexpression_assigner<xtensor_expression_tag>
    {};

    template <>
    struct xsparse_assigner<xsparse_expression_tag, extension::xsparse_assign_tag>
    {
        template <class E1, class E2>
        static void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            e1.derived_cast().resize(e2.derived_cast().shape());
            for(auto it = e2.derived_cast().nz_cbegin(); it != e2.derived_cast().nz_cend(); ++it)
            {
                e1.derived_cast().insert_element(it.index(), *it);
            }
        }
    };

    template <>
    struct xexpression_assigner<xsparse_expression_tag>
    {
        template <class E1, class E2>
        static void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            using e1_tag = xexpression_tag_t<E1>;
            using e2_tag = extension::get_assign_tag_t<E2>;

            xsparse_assigner<e1_tag, e2_tag>::assign_xexpression(e1, e2);
        }
    };

}

#endif
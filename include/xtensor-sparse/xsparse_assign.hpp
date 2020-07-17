#ifndef XSPARSE_ASSIGN_HPP
#define XSPARSE_ASSIGN_HPP

#include <xtensor/xassign.hpp>

#include "xsparse_expression.hpp"

namespace xt
{
    template <>
    class xexpression_assigner<xsparse_expression_tag>
        : public xexpression_assigner<xtensor_expression_tag>
    {
    };
}

#endif
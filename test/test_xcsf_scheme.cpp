#include "gtest/gtest.h"
#include <xtensor-sparse/xcsf_scheme.hpp>

namespace xt
{
    using index_type = svector<size_t>;
    using xcsf_scheme_type = xcsf_scheme<std::vector<index_type>,
                                         std::vector<index_type>,
                                         std::vector<double>>;
    TEST(xcsf_scheme, insert)
    {
        xcsf_scheme_type scheme;
        scheme.insert_element({0, 2}, 2.5);
        EXPECT_EQ(scheme.coordinate()[0], index_type({0}));
        EXPECT_EQ(scheme.coordinate()[1], index_type({2}));
    }
}
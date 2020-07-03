#include "gtest/gtest.h"
#include "test_common_macros.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
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

    TEST(xcsf_scheme, find)
    {
        xcsf_scheme_type scheme;
        scheme.insert_element({0, 2}, 2.5);
        scheme.insert_element({1, 5}, 8.2);

        auto elem = scheme.find_element({0, 3});
        EXPECT_EQ(elem, nullptr);
        elem = scheme.find_element({0, 2});
        EXPECT_EQ(*elem, 2.5);
        elem = scheme.find_element({1, 5});
        EXPECT_EQ(*elem, 8.2);
    }

    TEST(xcsf_scheme, remove)
    {
        xcsf_scheme_type scheme;
        scheme.insert_element({0, 2}, 2.5);
        scheme.insert_element({1, 5}, 8.2);

        scheme.remove_element({0, 2});
        EXPECT_EQ(scheme.storage().size(), 2);
        EXPECT_EQ(scheme.storage()[0], 0.);
        EXPECT_EQ(scheme.storage()[1], 8.2);

        scheme.remove_element({0, 0});
        EXPECT_EQ(scheme.storage().size(), 2);
        EXPECT_EQ(scheme.storage()[0], 0.);
        EXPECT_EQ(scheme.storage()[1], 8.2);
    }

    TEST(xcsf_scheme, update_entries)
    {
        xcsf_scheme_type scheme;
        scheme.insert_element({0}, 2.5);
        scheme.insert_element({5}, 8.2);
        scheme.insert_element({1}, 3.1);
        scheme.insert_element({8}, 6.7);

        svector<std::size_t> strides_1d{1};
        svector<std::size_t> strides_2d{5, 1};
        svector<std::size_t> strides_3d{6, 3, 1};

        scheme.update_entries(strides_1d, strides_2d);
        EXPECT_EQ(scheme.position().size(), 2);
        EXPECT_EQ(scheme.coordinate().size(), 2);
        EXPECT_EQ(scheme.position()[0], index_type({0, 2}));
        EXPECT_EQ(scheme.coordinate()[0], index_type({0, 1}));
        EXPECT_EQ(scheme.position()[1], index_type({0, 2, 4}));
        EXPECT_EQ(scheme.coordinate()[1], index_type({0, 1, 0, 3}));

        scheme.update_entries(strides_2d, strides_3d);
        EXPECT_EQ(scheme.position().size(), 3);
        EXPECT_EQ(scheme.coordinate().size(), 3);
        EXPECT_EQ(scheme.position()[0], index_type({0, 2}));
        EXPECT_EQ(scheme.coordinate()[0], index_type({0, 1}));
        EXPECT_EQ(scheme.position()[1], index_type({0, 2, 3}));
        EXPECT_EQ(scheme.coordinate()[1], index_type({0, 1, 0}));
        EXPECT_EQ(scheme.position()[2], index_type({0, 2, 3, 4}));
        EXPECT_EQ(scheme.coordinate()[2], index_type({0, 1, 2, 2}));

        scheme.update_entries(strides_3d, strides_1d);
        EXPECT_EQ(scheme.position().size(), 1);
        EXPECT_EQ(scheme.coordinate().size(), 1);
        EXPECT_EQ(scheme.position()[0], index_type({0, 4}));
        EXPECT_EQ(scheme.coordinate()[0], index_type({0, 1, 5, 8}));
    }
}
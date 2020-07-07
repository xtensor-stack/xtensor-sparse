#include "gtest/gtest.h"
#include <xtensor-sparse/xcoo_scheme.hpp>

namespace xt
{
    using index_type = svector<size_t>;
    using xcoo_scheme_type = xcoo_scheme<std::array<size_t, 2>,
                                         std::vector<index_type>,
                                         std::vector<double>>;
    xcoo_scheme_type make_coo_scheme()
    {
        xcoo_scheme_type scheme;
        scheme.insert_element({0, 2}, 2.5);
        scheme.insert_element({1, 1}, 3.0);
        scheme.insert_element({0, 4}, 1.7);
        scheme.insert_element({2, 7}, 5.4);
        return scheme;
    }

    TEST(xcoo_scheme, insert_element)
    {
        auto scheme = make_coo_scheme();

        EXPECT_EQ(scheme.coordinate()[0], index_type({0, 2}));
        EXPECT_EQ(scheme.coordinate()[1], index_type({0, 4}));
        EXPECT_EQ(scheme.coordinate()[2], index_type({1, 1}));
        EXPECT_EQ(scheme.coordinate()[3], index_type({2, 7}));

        EXPECT_EQ(scheme.storage()[0], 2.5);
        EXPECT_EQ(scheme.storage()[1], 1.7);
        EXPECT_EQ(scheme.storage()[2], 3.0);
        EXPECT_EQ(scheme.storage()[3], 5.4);

        EXPECT_EQ(scheme.position()[0], 0u);
        EXPECT_EQ(scheme.position()[1], 4u);
    }

    TEST(xcoo_scheme, find_element)
    {
        auto scheme = make_coo_scheme();

        auto p1 = scheme.find_element({0, 2});
        auto p2 = scheme.find_element({0, 4});
        auto p3 = scheme.find_element({1, 1});
        auto p4 = scheme.find_element({2, 2});
        auto p5 = scheme.find_element({2, 7});

        EXPECT_EQ(p1, &(scheme.storage()[0]));
        EXPECT_EQ(p2, &(scheme.storage()[1]));
        EXPECT_EQ(p3, &(scheme.storage()[2]));
        EXPECT_EQ(p4, nullptr);
        EXPECT_EQ(p5, &(scheme.storage()[3]));
    }

    TEST(xcoo_scheme, remove_element)
    {
        auto scheme = make_coo_scheme();
        scheme.remove_element({0, 4});

        EXPECT_EQ(scheme.coordinate()[0], index_type({0, 2}));
        EXPECT_EQ(scheme.coordinate()[1], index_type({1, 1}));
        EXPECT_EQ(scheme.coordinate()[2], index_type({2, 7}));

        EXPECT_EQ(scheme.storage()[0], 2.5);
        EXPECT_EQ(scheme.storage()[1], 3.0);
        EXPECT_EQ(scheme.storage()[2], 5.4);

        EXPECT_EQ(scheme.position()[0], 0u);
        EXPECT_EQ(scheme.position()[1], 3u);
    }

    TEST(xcoo_scheme, update_entries)
    {
        auto scheme = make_coo_scheme();
        std::vector<size_t> old_strides = {8, 1};
        std::vector<size_t> new_strides = {8, 4, 1};
        scheme.update_entries(old_strides, new_strides);

        EXPECT_EQ(scheme.coordinate()[0], index_type({0, 0, 2}));
        EXPECT_EQ(scheme.coordinate()[1], index_type({0, 1, 0}));
        EXPECT_EQ(scheme.coordinate()[2], index_type({1, 0, 1}));
        EXPECT_EQ(scheme.coordinate()[3], index_type({2, 1, 3}));
    }

    template <class S>
    class coo_scheme_iterator : public ::testing::Test
    {
    public:

        using schmee_type = S;
    };

    using coo_iterator_test_types = ::testing::Types<xcoo_scheme_type, const xcoo_scheme_type>;
    TYPED_TEST_SUITE(coo_scheme_iterator, coo_iterator_test_types);

    TYPED_TEST(coo_scheme_iterator, increment)
    {
        TypeParam scheme = make_coo_scheme();
        auto it = scheme.begin();
        EXPECT_EQ(*it, 2.5);
        EXPECT_EQ(it.index(), index_type({0, 2}));
        ++it;
        EXPECT_EQ(*it, 1.7);
        EXPECT_EQ(it.index(), index_type({0, 4}));
        ++it;
        EXPECT_EQ(*it, 3.0);
        EXPECT_EQ(it.index(), index_type({1, 1}));
        ++it;
        EXPECT_EQ(*it, 5.4);
        EXPECT_EQ(it.index(), index_type({2, 7}));
        ++it;
        EXPECT_EQ(it, scheme.end());

        auto it2 = scheme.begin();
        it2 += 2;
        EXPECT_EQ(*it2, 3.0);
        EXPECT_EQ(it2.index(), index_type({1, 1}));
    }

    TYPED_TEST(coo_scheme_iterator, decrement)
    {
        TypeParam scheme = make_coo_scheme();
        auto it = scheme.end();
        --it;
        EXPECT_EQ(*it, 5.4);
        EXPECT_EQ(it.index(), index_type({2, 7}));
        --it;
        EXPECT_EQ(*it, 3.0);
        EXPECT_EQ(it.index(), index_type({1, 1}));
        --it;
        EXPECT_EQ(*it, 1.7);
        EXPECT_EQ(it.index(), index_type({0, 4}));
        --it;
        EXPECT_EQ(*it, 2.5);
        EXPECT_EQ(it.index(), index_type({0, 2}));
        EXPECT_EQ(it, scheme.begin());

        auto it2 = scheme.end();
        it2 -= 2;
        EXPECT_EQ(*it2, 3.0);
        EXPECT_EQ(it2.index(), index_type({1, 1}));
    }
}


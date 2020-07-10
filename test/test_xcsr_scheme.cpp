#include "gtest/gtest.h"

#include "xtensor-sparse/xcsr_scheme.hpp"

namespace xt
{
    using index_type = std::size_t;
    using xcsr_scheme_type = xcsr_scheme<std::vector<index_type>,
                                         std::vector<index_type>,
                                         std::vector<double>>;
    TEST(xcsr_scheme, insert)
    {
        xcsr_scheme_type scheme(5);
        scheme.insert_element({0, 2}, 2.5);
        scheme.insert_element({1, 5}, 8.2);
        scheme.insert_element({0, 8}, 5.8);

        EXPECT_EQ(scheme.coordinate()[0], index_type(2));
        EXPECT_EQ(scheme.storage()[0], 2.5);
    }

    TEST(xcsr_scheme, find)
    {
        xcsr_scheme_type scheme(5);
        scheme.insert_element({0, 2}, 2.5);
        scheme.insert_element({1, 5}, 8.2);

        auto elem = scheme.find_element({0, 3});
        EXPECT_EQ(elem, nullptr);
        elem = scheme.find_element({0, 2});
        EXPECT_EQ(*elem, 2.5);
        elem = scheme.find_element({1, 5});
        EXPECT_EQ(*elem, 8.2);
    }

    TEST(xcsr_scheme, remove)
    {
        xcsr_scheme_type scheme(5);
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

    TEST(xcsr_scheme, update_entries)
    {
        svector<std::size_t> shape{10, 10};
        xcsr_scheme_type scheme(shape[0]);
        scheme.insert_element({0, 4}, 2.5);
        scheme.insert_element({2, 5}, 8.2);
        scheme.insert_element({1, 1}, 3.1);
        scheme.insert_element({3, 8}, 6.7);

        svector<std::size_t> old_strides{10, 1};
        svector<std::size_t> new_strides{5, 1};
        std::array<std::size_t, 2> new_shape{{20, 5}};

        scheme.update_entries(old_strides, new_strides, new_shape);
        EXPECT_EQ(scheme.position().size(), new_shape[0] + 1);
        EXPECT_EQ(scheme.coordinate().size(), 4);
        EXPECT_EQ(scheme.position(), svector<std::size_t>({0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}));
        EXPECT_EQ(scheme.coordinate(), svector<std::size_t>({4, 1, 0, 3}));
        EXPECT_EQ(scheme.storage(), svector<double>({2.5, 3.1, 8.2, 6.7}));

        scheme.update_entries(new_strides, old_strides, shape);
        EXPECT_EQ(scheme.position().size(), shape[0] + 1);
        EXPECT_EQ(scheme.coordinate().size(), 4);
        EXPECT_EQ(scheme.position(), svector<std::size_t>({0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4}));
        EXPECT_EQ(scheme.coordinate(), svector<std::size_t>({4, 1, 5, 8}));
        EXPECT_EQ(scheme.storage(), svector<double>({2.5, 3.1, 8.2, 6.7}));
    }

    TEST(xcsr_scheme, iterator_forward)
    {
        svector<std::size_t> shape{10, 10};
        xcsr_scheme_type scheme(shape[0]);
        scheme.insert_element({0, 4}, 2.5);
        scheme.insert_element({2, 5}, 8.2);
        scheme.insert_element({1, 1}, 3.1);
        scheme.insert_element({3, 8}, 6.7);

        auto it = scheme.nz_begin();
        std::array<std::size_t, 2> expected{{0, 4}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 2.5);
        ++it;
        expected = {{1, 1}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 3.1);
        ++it;
        expected = {{2, 5}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 8.2);
        ++it;
        expected = {{3, 8}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 6.7);
        ++it;

        svector<std::size_t> old_strides{10, 1};
        svector<std::size_t> new_strides{5, 1};
        svector<std::size_t> new_shape{20, 5};

        scheme.update_entries(old_strides, new_strides, new_shape);
        it = scheme.nz_begin();
        expected = {{0, 4}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 2.5);
        ++it;
        expected = {{2, 1}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 3.1);
        ++it;
        expected = {{5, 0}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 8.2);
        ++it;
        expected = {{7, 3}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 6.7);
        ++it;

        it = scheme.nz_begin();
        it += 2;
        expected = {{5, 0}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 8.2);
    }

    TEST(xcsr_scheme, iterator_backward)
    {
        svector<std::size_t> shape{10, 10};
        xcsr_scheme_type scheme(shape[0]);
        scheme.insert_element({0, 4}, 2.5);
        scheme.insert_element({2, 5}, 8.2);
        scheme.insert_element({1, 1}, 3.1);
        scheme.insert_element({3, 8}, 6.7);

        auto it = scheme.nz_end();
        --it;
        std::array<std::size_t, 2> expected{{3, 8}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 6.7);
        --it;
        expected = {{2, 5}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 8.2);
        --it;
        expected = {{1, 1}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 3.1);
        --it;
        expected = {{0, 4}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 2.5);
        --it;

        svector<std::size_t> old_strides{10, 1};
        svector<std::size_t> new_strides{5, 1};
        svector<std::size_t> new_shape{20, 5};

        scheme.update_entries(old_strides, new_strides, new_shape);
        it = scheme.nz_end();
        --it;
        expected = {{7, 3}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 6.7);
        --it;
        expected = {{5, 0}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 8.2);
        --it;
        expected = {{2, 1}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 3.1);
        --it;
        expected = {{0, 4}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 2.5);
        --it;

        it = scheme.nz_end();
        it -= 3;
        expected = {{2, 1}};
        EXPECT_EQ(it.index(), expected);
        EXPECT_EQ(*it, 3.1);
    }
}

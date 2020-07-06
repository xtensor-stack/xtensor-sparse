#include "gtest/gtest.h"
#include <iostream>
#include <map>
#include <xtensor-sparse/xsparse_reference.hpp>

namespace xt
{
    class sparse_tester
    {
    public:

        using map_type = std::map<int, double>;
        using value_type = double;
        using pointer = double*;
        using const_reference = const double&;
        using index_type = int;

        sparse_tester();
        const map_type& data() const;

        pointer find_element(const index_type& index);
        void insert_element(const index_type& index, const_reference v);
        void remove_element(const index_type& index);

    private:

        map_type m_data;
    };

    using sparse_ref = xsparse_reference<sparse_tester>;

    sparse_tester::sparse_tester()
        : m_data(map_type({{1, 2.0}, {3, 4.5}, {9, 2.7}}))
    {
    }

    auto sparse_tester::data() const -> const map_type&
    {
        return m_data;
    }

    auto sparse_tester::find_element(const index_type& index) -> pointer
    {
        auto it = m_data.find(index);
        return it != m_data.end() ? &(it->second) : nullptr;
    }

    void sparse_tester::insert_element(const index_type& index, const_reference v)
    {
        m_data.insert({index, v});
    }

    void sparse_tester::remove_element(const index_type& index)
    {
        m_data.erase(index);
    }

    TEST(xsparse_reference, assign_semantic)
    {
        sparse_tester t;

        sparse_ref r1(t, 1, 2.);
        sparse_ref r2(t, 3, 4.5);
        r1 = r2;
        sparse_tester::map_type exp1 = {{1, 4.5}, {3, 4.5}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp1);

        sparse_ref r3(t, 9, 2.7);
        r2 = std::move(r3);
        sparse_tester::map_type exp2 = {{1, 4.5}, {3, 2.7}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp2);
    }

    TEST(xsparse_reference, assign)
    {
        sparse_tester t;
        
        sparse_ref r1(t, 1, 2.);
        r1 = 2.5;
        sparse_tester::map_type exp1 = {{1, 2.5}, {3, 4.5}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp1);

        sparse_ref r2(t, 4, 0);
        r2 = 3.2;
        sparse_tester::map_type exp2 = {{1, 2.5}, {3, 4.5}, {4, 3.2}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp2);

        sparse_ref r3(t, 3, 4.5);
        r3 = 0.;
        sparse_tester::map_type exp3 = {{1, 2.5}, {4, 3.2}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp3);
    }

    TEST(xsparse_reference, plus_assign)
    {
        sparse_tester t;
        
        sparse_ref r1(t, 1, 2.);
        r1 += 2.5;
        sparse_tester::map_type exp1 = {{1, 4.5}, {3, 4.5}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp1);

        sparse_ref r2(t, 4, 0);
        r2 += 3.2;
        sparse_tester::map_type exp2 = {{1, 4.5}, {3, 4.5}, {4, 3.2}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp2);

        sparse_ref r3(t, 3, 4.5);
        r3 += -4.5;
        sparse_tester::map_type exp3 = {{1, 4.5}, {4, 3.2}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp3);
    }

    TEST(xsparse_reference, minus_assign)
    {
        sparse_tester t;
        
        sparse_ref r1(t, 1, 2.);
        r1 -= 1.5;
        sparse_tester::map_type exp1 = {{1, 0.5}, {3, 4.5}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp1);

        sparse_ref r2(t, 4, 0);
        r2 -= -3.2;
        sparse_tester::map_type exp2 = {{1, 0.5}, {3, 4.5}, {4, 3.2}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp2);

        sparse_ref r3(t, 3, 4.5);
        r3 -= 4.5;
        sparse_tester::map_type exp3 = {{1, 0.5}, {4, 3.2}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp3);
    }

    TEST(xsparse_reference, mul_assign)
    {
        sparse_tester t;
        
        sparse_ref r1(t, 1, 2.);
        r1 *= 1.5;
        sparse_tester::map_type exp1 = {{1, 3.}, {3, 4.5}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp1);

        sparse_ref r3(t, 3, 4.5);
        r3 *= 0.;
        sparse_tester::map_type exp3 = {{1, 3.}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp3);
    }

    TEST(xsparse_reference, div_assign)
    {
        sparse_tester t;
        
        sparse_ref r1(t, 1, 2.);
        r1 /= 2.;
        sparse_tester::map_type exp1 = {{1, 1.}, {3, 4.5}, {9, 2.7}};
        EXPECT_EQ(t.data(), exp1);
    }

    TEST(xsparse_reference, conversion)
    {
        sparse_tester t;
        sparse_ref r1(t, 1, 2.);

        double d = r1;
        EXPECT_EQ(d, 2.);
    }

}

#ifndef XSPARSE_SPARSE_REFERENCE_HPP
#define XSPARSE_SPARSE_REFERENCE_HPP

namespace xt
{

    /*********************
     * xsparse_reference *
     *********************/

    template <class C>
    class xsparse_reference
    {
    public:

        using self_type = xsparse_reference<C>;
        using container_type = C;
        using value_type = typename container_type::value_type;
        using const_reference = typename container_type::const_reference;
        using index_type = typename container_type::index_type;

        xsparse_reference(container_type& c, index_type index, const_reference value);
        ~xsparse_reference() = default;

        xsparse_reference(const self_type&) = default;
        xsparse_reference(self_type&&) = default;

        self_type& operator=(const self_type&);
        self_type& operator=(self_type&&);

        template <class T>
        self_type& operator=(const T&);

        template <class T>
        self_type& operator+=(const T&);

        template <class T>
        self_type& operator-=(const T&);

        template <class T>
        self_type& operator*=(const T&);

        template <class T>
        self_type& operator/=(const T&);

        operator const_reference() const;

    private:

        using pointer = typename container_type::true_pointer;
        void update_value(const_reference value);

        container_type& m_container;
        index_type m_index;
        value_type m_value;
    };

    /************************************
     * xsparse_reference implementation *
     ************************************/

    template <class C>
    inline xsparse_reference<C>::xsparse_reference(container_type& c, index_type index, const_reference value)
        : m_container(c), m_index(std::move(index)), m_value(value)
    {
    }

    template <class C>
    inline auto xsparse_reference<C>::operator=(const self_type& rhs) -> self_type&
    {
        update_value(rhs.m_value);
        return *this;
    }

    template <class C>
    inline auto xsparse_reference<C>::operator=(self_type&& rhs) -> self_type&
    {
        update_value(rhs.m_value);
        return *this;
    }

    template <class C>
    template <class T>
    inline auto xsparse_reference<C>::operator=(const T& rhs) -> self_type&
    {
        update_value(rhs);
        return *this;
    }

    template <class C>
    template <class T>
    inline auto xsparse_reference<C>::operator+=(const T& rhs) -> self_type&
    {
        update_value(m_value + rhs);
        return *this;
    }

    template <class C>
    template <class T>
    inline auto xsparse_reference<C>::operator-=(const T& rhs) -> self_type&
    {
        update_value(m_value - rhs);
        return *this;
    }

    template <class C>
    template <class T>
    inline auto xsparse_reference<C>::operator*=(const T& rhs) -> self_type&
    {
        update_value(m_value * rhs);
        return *this;
    }

    template <class C>
    template <class T>
    inline auto xsparse_reference<C>::operator/=(const T& rhs) -> self_type&
    {
        update_value(m_value / rhs);
        return *this;
    }

    template <class C>
    inline xsparse_reference<C>::operator const_reference() const
    {
        return m_value;
    }

    template <class C>
    inline void xsparse_reference<C>::update_value(const_reference value)
    {
        pointer p = m_container.find_element(m_index);
        if(p != nullptr)
        {
            if(value == value_type(0))
            {
                m_container.remove_element(m_index);
            }
            else
            {
                *p = value;
            }
        }
        else
        {
            m_container.insert_element(m_index, value);
        }
        m_value = value;
    }
}

#endif


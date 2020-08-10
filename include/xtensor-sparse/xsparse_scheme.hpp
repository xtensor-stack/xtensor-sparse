#ifndef XSPARSE_SCHEME_HPP
#define XSPARSE_SCHEME_HPP

#include <xtl/xany.hpp>
#include <xtensor/xstorage.hpp>

namespace xt
{

    /***********************************************************************
     * xsparse_polymorphic_scheme_nz_iterator as a bridge for type erasure *
     ***********************************************************************/

    template <class T>
    class xsparse_abstract_scheme_nz_iterator;

    template <class T>
    class xsparse_polymorphic_scheme_nz_iterator
    {
    public:

        using self_type = xsparse_polymorphic_scheme_nz_iterator<T>;
        using abstract_iterator = xsparse_abstract_scheme_nz_iterator<T>;
        using index_type = xtl::any;
        using value_type = T;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        xsparse_polymorphic_scheme_nz_iterator(abstract_iterator *it);
        ~xsparse_polymorphic_scheme_nz_iterator();

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;
        const index_type& index() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

    private:
        abstract_iterator *m_it = nullptr;
    };

    template <class T>
    bool operator == (const xsparse_polymorphic_scheme_nz_iterator<T>& lhs,
                   const xsparse_polymorphic_scheme_nz_iterator<T>& rhs);

    template <class T>
    bool operator < (const xsparse_polymorphic_scheme_nz_iterator<T>& lhs,
                  const xsparse_polymorphic_scheme_nz_iterator<T>& rhs);

    /***************************************************************************
     * xsparse_abstract_scheme_nz_iterator as top-level class for type erasure *
     ***************************************************************************/

    template <class T>
    class xsparse_abstract_scheme_nz_iterator
    {
    public:

        using self_type = xsparse_abstract_scheme_nz_iterator<T>;
        using index_type = xtl::any;
        using value_type = T;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        virtual ~xsparse_abstract_scheme_nz_iterator() = default;

        virtual reference operator*() const = 0;
        virtual pointer operator->() const = 0;
        virtual const index_type& index() const = 0;

        virtual bool equal(const self_type& rhs) const = 0;
        virtual bool less_than(const self_type& rhs) const = 0;

        virtual difference_type distance(const self_type& rhs) const = 0;

        virtual void advance(void) = 0;
        virtual void  rewind(void) = 0;
        virtual void advance(difference_type n) = 0;
        virtual void  rewind(difference_type n) = 0;
    };

    /******************************************************************
     * xsparse_crtp_scheme_nz_iterator as base class for type erasure *
     ******************************************************************/

    template <class T, class D>
    class xsparse_crtp_scheme_nz_iterator : public xsparse_abstract_scheme_nz_iterator<T>
    {
    public:

        using derived_type = D;

        using self_type = xsparse_crtp_scheme_nz_iterator<T, D>;
        using index_type = xtl::any;
        using value_type = T;
        using reference = value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        const index_type& index() const final;

        bool equal(const self_type& rhs) const final;
        bool less_than(const self_type& rhs) const final;

        difference_type distance(const self_type& rhs) const final;

        void advance(void) final;
        void  rewind(void) final;
        void advance(difference_type n) final;
        void  rewind(difference_type n) final;

    private:

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const & noexcept;
        derived_type derived_cast() && noexcept;

        index_type m_index;
    };

    /************************************************
     * xsparse_coo_scheme_nz_iterator as an example *
     ************************************************/

    namespace detail
    {
        template <class scheme>
        struct xsparse_coo_scheme_storage_type
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::iterator;
        };

        template <class scheme>
        struct xsparse_coo_scheme_storage_type<const scheme>
        {
            using storage_type = typename scheme::storage_type;
            using value_iterator = typename storage_type::const_iterator;
        };

        template <class scheme>
        struct xsparse_coo_scheme_nz_iterator_types : xsparse_coo_scheme_storage_type<scheme>
        {
            using base_type = xsparse_coo_scheme_storage_type<scheme>;
            using index_type = typename scheme::index_type;
            using coordinate_type = typename scheme::coordinate_type;
            using coordinate_iterator = typename coordinate_type::const_iterator;
            using value_iterator = typename base_type::value_iterator;
            using value_type = typename value_iterator::value_type;
            using reference = typename value_iterator::reference;
            using pointer = typename value_iterator::pointer;
            using difference_type = typename value_iterator::difference_type;
        };
    }

    template <class scheme>
    class xsparse_coo_scheme_nz_iterator : public xsparse_crtp_scheme_nz_iterator<typename scheme::value_type,
                                                                                  xsparse_coo_scheme_nz_iterator<scheme>>,
                                           xtl::xrandom_access_iterator_base3<xsparse_coo_scheme_nz_iterator<scheme>,
                                                                              detail::xsparse_coo_scheme_nz_iterator_types<scheme>>
    {
    public:

        using self_type = xsparse_coo_scheme_nz_iterator<scheme>;
        using scheme_type = scheme;
        using iterator_types = detail::xsparse_coo_scheme_nz_iterator_types<scheme>;
        using index_type = typename iterator_types::index_type;
        using coordinate_type = typename iterator_types::coordinate_type;
        using coordinate_iterator = typename iterator_types::coordinate_iterator;
        using value_iterator = typename iterator_types::value_iterator;
        using value_type = typename iterator_types::value_type;
        using reference = typename iterator_types::reference;
        using pointer = typename iterator_types::pointer;
        using difference_type = typename iterator_types::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        xsparse_coo_scheme_nz_iterator() = default;
        xsparse_coo_scheme_nz_iterator(scheme& s, coordinate_iterator cit, value_iterator vit);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;
        const index_type& index() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

    private:

        scheme_type* p_scheme = nullptr;
        coordinate_iterator m_cit;
        value_iterator m_vit;
    };

    template <class S>
    bool operator==(const xsparse_coo_scheme_nz_iterator<S>& lhs,
                 const xsparse_coo_scheme_nz_iterator<S>& rhs);

    template <class S>
    bool operator<(const xsparse_coo_scheme_nz_iterator<S>& lhs,
                const xsparse_coo_scheme_nz_iterator<S>& rhs);

    /*********************************************************
     * xsparse_polymorphic_scheme_nz_iterator implementation *
     *********************************************************/

    template <class T>
    inline xsparse_polymorphic_scheme_nz_iterator<T>::xsparse_polymorphic_scheme_nz_iterator(abstract_iterator *it) : m_it(it)
    {
    }

    template <class T>
    inline xsparse_polymorphic_scheme_nz_iterator<T>::~xsparse_polymorphic_scheme_nz_iterator()
    {
        if (m_it)
            delete m_it;
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::operator++() -> self_type&
    {
        m_it->advance();
        return *this;
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::operator--() -> self_type&
    {
        m_it->rewind();
        return *this;
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::operator+=(difference_type n) -> self_type&
    {
        m_it->advance(n);
        return *this;
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::operator-=(difference_type n) -> self_type&
    {
        m_it->rewind(n);
        return *this;
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_it->distance(rhs);
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::operator*() const -> reference
    {
        return m_it->reference();
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::operator->() const -> pointer
    {
        return m_it->pointer();
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme_nz_iterator<T>::index() const -> const index_type&
    {
        return m_it->index();
    }

    template <class T>
    inline bool xsparse_polymorphic_scheme_nz_iterator<T>::equal(const self_type& rhs) const
    {
        return m_it->equal(*(rhs->m_it));
    }

    template <class T>
    inline bool xsparse_polymorphic_scheme_nz_iterator<T>::less_than(const self_type& rhs) const
    {
        return m_it->less_than(*(rhs->m_it));
    }

    template <class T>
    inline bool operator == (const xsparse_polymorphic_scheme_nz_iterator<T>& lhs,
                          const xsparse_polymorphic_scheme_nz_iterator<T>& rhs)
    {
        return lhs->equal(rhs);
    }

    template <class T>
    inline bool operator < (const xsparse_polymorphic_scheme_nz_iterator<T>& lhs,
                         const xsparse_polymorphic_scheme_nz_iterator<T>& rhs)
    {
        return lhs->less_than(rhs);
    }

    /**************************************************
     * xsparse_crtp_scheme_nz_iterator implementation *
     **************************************************/

    template <class T, class D>
    inline auto xsparse_crtp_scheme_nz_iterator<T, D>::index() const -> const index_type&
    {
        m_index = this->derived_cast().index();
        return m_index;
    }

    template <class T, class D>
    inline bool xsparse_crtp_scheme_nz_iterator<T, D>::equal(const self_type& rhs) const
    {
        return this->derived_cast() == static_cast<const derived_type&>(rhs);
    }

    template <class T, class D>
    inline bool xsparse_crtp_scheme_nz_iterator<T, D>::less_than(const self_type& rhs) const
    {
        return this->derived_cast() < static_cast<const derived_type&>(rhs);
    }

    template <class T, class D>
    inline auto xsparse_crtp_scheme_nz_iterator<T, D>::distance(const self_type& rhs) const -> difference_type
    {
        auto self = this->derived_cast();
        auto other = static_cast<const derived_type&>(rhs);

        auto diff = self - other;
        return (difference_type)(diff);
    }

    template <class T, class D>
    inline void xsparse_crtp_scheme_nz_iterator<T, D>::advance(void)
    {
        ++(this->derived_cast());
    }

    template <class T, class D>
    inline void xsparse_crtp_scheme_nz_iterator<T, D>::rewind(void)
    {
        --(this->derived_cast());
    }

    template <class T, class D>
    inline void xsparse_crtp_scheme_nz_iterator<T, D>::advance(difference_type n)
    {
        (this->derived_cast()) += n;
    }

    template <class T, class D>
    inline void xsparse_crtp_scheme_nz_iterator<T, D>::rewind(difference_type n)
    {
        (this->derived_cast()) -= n;
    }

    template <class T, class D>
    inline auto xsparse_crtp_scheme_nz_iterator<T, D>::derived_cast() & noexcept -> derived_type&
    {
        return static_cast<derived_type&>(*this);
    }

    template <class T, class D>
    inline auto xsparse_crtp_scheme_nz_iterator<T, D>::derived_cast() const & noexcept -> const derived_type&
    {
        return static_cast<const derived_type&>(*this);
    }

    /*************************************************
     * xsparse_coo_scheme_nz_iterator implementation *
     *************************************************/

    template <class S>
    inline xsparse_coo_scheme_nz_iterator<S>::xsparse_coo_scheme_nz_iterator(S& s, coordinate_iterator cit, value_iterator vit)
        : p_scheme(&s)
        , m_cit(cit)
        , m_vit(vit)
    {
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::operator++() -> self_type&
    {
        ++m_cit;
        ++m_vit;
        return *this;
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::operator--() -> self_type&
    {
        --m_cit;
        --m_vit;
        return *this;
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::operator+=(difference_type n)  -> self_type&
    {
        m_cit += n;
        m_vit += n;
        return *this;
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::operator-=(difference_type n)  -> self_type&
    {
        m_cit -= n;
        m_vit -= n;
        return *this;
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_cit - rhs.m_cit;
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::operator*() const -> reference
    {
        return *m_vit;
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::operator->() const -> pointer
    {
        return &(*m_vit);
    }

    template <class S>
    inline auto xsparse_coo_scheme_nz_iterator<S>::index() const  -> const index_type&
    {
        return *m_cit;
    }

    template <class S>
    inline bool xsparse_coo_scheme_nz_iterator<S>::equal(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_cit == rhs.m_cit && m_vit == rhs.m_vit;
    }

    template <class S>
    inline bool xsparse_coo_scheme_nz_iterator<S>::less_than(const self_type& rhs) const
    {
        return p_scheme == rhs.p_scheme && m_cit < rhs.m_cit && m_vit < rhs.m_vit;
    }

    template <class S>
    inline bool operator == (const xsparse_coo_scheme_nz_iterator<S>& lhs,
                          const xsparse_coo_scheme_nz_iterator<S>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class S>
    inline bool operator < (const xsparse_coo_scheme_nz_iterator<S>& lhs,
                         const xsparse_coo_scheme_nz_iterator<S>& rhs)
    {
        return lhs.less_than(rhs);
    }

    /***********************************************************
     * xsparse_polymorphic_scheme as a bridge for type erasure *
     ***********************************************************/

    template <class T>
    class xsparse_abstract_scheme;

    template <class T>
    class xsparse_polymorphic_scheme
    {
    public:

        using self_type = xsparse_polymorphic_scheme<T>;
        using index_type = xtl::any;

        using value_type = T;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        using size_type = std::size_t;
        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;
        using inner_shape_type = shape_type;

        using nz_iterator = xsparse_polymorphic_scheme_nz_iterator<self_type>;
        using const_nz_iterator = xsparse_polymorphic_scheme_nz_iterator<const self_type>;

        xsparse_polymorphic_scheme();
        xsparse_polymorphic_scheme(xsparse_abstract_scheme<T> *scheme);
        ~xsparse_polymorphic_scheme();

        pointer find_element(const index_type& index);
        const_pointer find_element(const index_type& index) const;
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        void update_entries(const strides_type& old_strides,
                            const strides_type& new_strides,
                            const shape_type& new_shape);

        nz_iterator nz_begin();
        nz_iterator nz_end();
        const_nz_iterator nz_begin() const;
        const_nz_iterator nz_end() const;
        const_nz_iterator nz_cbegin() const;
        const_nz_iterator nz_cend() const;

    private:
        class xsparse_abstract_scheme<T> *m_scheme = nullptr;
    };

    /***********************************************************
     * xsparse_abstract_scheme as base class for type erasure  *
     ***********************************************************/

    template <class T>
    class xsparse_abstract_scheme
    {
    public:

        using self_type = xsparse_abstract_scheme<T>;
        using index_type = xtl::any;

        using value_type = T;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        using size_type = std::size_t;
        using shape_type = svector<size_type>;
        using strides_type = svector<size_type>;
        using inner_shape_type = shape_type;

        using nz_iterator = xsparse_polymorphic_scheme_nz_iterator<self_type>;
        using const_nz_iterator = xsparse_polymorphic_scheme_nz_iterator<const self_type>;


        virtual ~xsparse_abstract_scheme = default;

        virtual pointer find_element(const index_type& index) = 0;
        virtual const_pointer find_element(const index_type& index) const = 0;
        virtual void insert_element(const index_type& index, const_reference value) = 0;
        virtual void remove_element(const index_type& index) = 0;

        virtual void update_entries(const strides_type& old_strides,
                                    const strides_type& new_strides,
                                    const shape_type& new_shape) = 0;

        virtual nz_iterator nz_begin() = 0;
        virtual nz_iterator nz_end() = 0;
        virtual const_nz_iterator nz_begin() const = 0;
        virtual const_nz_iterator nz_end() const = 0;
        virtual const_nz_iterator nz_cbegin() const = 0;
        virtual const_nz_iterator nz_cend() const = 0;
    };

    /**********************
     * xsparse_coo_scheme *
     **********************/

    template <class P, class C, class ST, class IT = svector<std::size_t>>
    class xsparse_coo_scheme
    {
    public:

        using self_type = xsparse_coo_scheme<P, C, ST, IT>;
        using position_type = P;
        using coordinate_type = C;
        using storage_type = ST;
        using index_type = IT;

        using value_type = typename storage_type::value_type;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;
        using const_pointer = typename storage_type::const_pointer;

        using nz_iterator = xsparse_polymorphic_scheme_nz_iterator<self_type>;
        using const_nz_iterator = xsparse_polymorphic_scheme_nz_iterator<const self_type>;

        using coo_nz_iterator = xsparse_coo_scheme_nz_iterator<self_type>;
        using coo_const_nz_iterator = xsparse_coo_scheme_nz_iterator<const self_type>;

        xsparse_coo_scheme();

        const position_type& position() const;
        const coordinate_type& coordinate() const;
        const storage_type& storage() const;


        pointer find_element(const index_type& index);
        const_pointer find_element(const index_type& index) const;
        void insert_element(const index_type& index, const_reference value);
        void remove_element(const index_type& index);

        template <class strides_type, class shape_type>
        void update_entries(const strides_type& old_strides,
                            const strides_type& new_strides,
                            const shape_type& new_shape);

        nz_iterator nz_begin();
        nz_iterator nz_end();
        const_nz_iterator nz_begin() const;
        const_nz_iterator nz_end() const;
        const_nz_iterator nz_cbegin() const;
        const_nz_iterator nz_cend() const;

    private:

        const_pointer find_element_impl(const index_type& index) const;

        position_type m_pos;
        coordinate_type m_coords;
        storage_type m_storage;

        friend class xsparse_coo_scheme_nz_iterator<self_type>;
        friend class xsparse_coo_scheme_nz_iterator<const self_type>;
    };


    /*********************************************
     * xsparse_polymorphic_scheme implementation *
     *********************************************/

    template <class T>
    inline xsparse_polymorphic_scheme<T>::xsparse_polymorphic_scheme()
    {
        // m_scheme = xt::scheme_policy().scheme();
    }

    template <class T>
    inline xsparse_polymorphic_scheme<T>::xsparse_polymorphic_scheme(xsparse_abstract_scheme<T> *scheme) : m_scheme(scheme)
    {

    }

    template <class T>
    inline xsparse_polymorphic_scheme<T>::~xsparse_polymorphic_scheme()
    {
        if (m_scheme)
            delete m_scheme;
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::find_element(const index_type& index) -> pointer
    {
        return m_scheme->find_element(index);
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::find_element(const index_type& index) const -> const_pointer
    {
        return m_scheme->find_element(index);
    }

    template <class T>
    inline void xsparse_polymorphic_scheme<T>::insert_element(const index_type& index, const_reference value)
    {
        m_scheme->insert_element(index, value);
    }

    template <class T>
    inline void xsparse_polymorphic_scheme<T>::remove_element(const index_type& index)
    {
        m_scheme->remove_element(index);
    }

    template <class T>
    inline void xsparse_polymorphic_scheme<T>::update_entries(const strides_type& old_strides,
                                                              const strides_type& new_strides,
                                                              const shape_type& new_shape)
    {
        m_scheme->update_entries(old_strides, new_strides, new_shape);
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::nz_begin() -> nz_iterator
    {
        return m_scheme->nz_begin();
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::nz_end() -> nz_iterator
    {
        return m_scheme->nz_end();
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::nz_begin() const -> const_nz_iterator
    {
        return m_scheme->nz_begin();
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::nz_end() const -> const_nz_iterator
    {
        return m_scheme->nz_end();
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::nz_cbegin() const -> const_nz_iterator
    {
        return m_scheme->nz_cbegin();
    }

    template <class T>
    inline auto xsparse_polymorphic_scheme<T>::nz_cend() const -> const_nz_iterator
    {
        return m_scheme->nz_cend();
    }

    /******************************************
     * xsparse_abstract_scheme implementation *
     ******************************************/

    template <class P, class C, class ST, class IT>
    inline xsparse_coo_scheme<P, C, ST, IT>::xsparse_coo_scheme()
        : m_pos(P{{0u, 0u}})


    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::position() const -> const position_type&
    {
        return m_pos;
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::coordinate() const -> const coordinate_type&
    {
        return m_coords;
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::storage() const -> const storage_type&
    {
        return m_storage;
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::find_element(const index_type& index) -> pointer
    {
        return const_cast<pointer>(find_element_impl(index));
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::find_element(const index_type& index) const -> const_pointer
    {
        return find_element_impl(index);
    }

    template <class P, class C, class ST, class IT>
    inline void xsparse_coo_scheme<P, C, ST, IT>::insert_element(const index_type& index, const_reference value)
    {
        auto it = std::upper_bound(m_coords.cbegin(), m_coords.cend(), index);
        if (it != m_coords.cend())
        {
            auto diff = std::distance(m_coords.cbegin(), it);
            m_coords.insert(it, index);
            m_storage.insert(m_storage.cbegin() + diff, value);
        }
        else
        {
            m_coords.push_back(index);
            m_storage.push_back(value);
        }
        ++m_pos.back();
    }

    template <class P, class C, class ST, class IT>
    inline void xsparse_coo_scheme<P, C, ST, IT>::remove_element(const index_type& index)
    {
        auto it = std::find(m_coords.begin(), m_coords.end(), index);
        if (it != m_coords.end())
        {
            auto diff = it - m_coords.begin();
            m_coords.erase(it);
            m_pos.back()--;
            m_storage.erase(m_storage.begin() + diff);
        }
    }

    template <class P, class C, class ST, class IT>
    template <class strides_type, class shape_type>
    inline void xsparse_coo_scheme<P, C, ST, IT>::update_entries(const strides_type& old_strides,
                                                                 const strides_type& new_strides,
                                                                 const shape_type&)
    {
        coordinate_type new_coords;

        for(auto& old_index: m_coords)
        {
            std::size_t offset = element_offset<std::size_t>(old_strides, old_index.cbegin(), old_index.cend());
            index_type new_index = unravel_from_strides(offset, new_strides);
            new_coords.push_back(new_index);
        }
        using std::swap;
        swap(m_coords, new_coords);
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::find_element_impl(const index_type& index) const -> const_pointer
    {
        auto it = std::find(m_coords.begin(), m_coords.end(), index);
        return it == m_coords.end() ? nullptr : &*(m_storage.begin() + (it - m_coords.begin()));
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::nz_begin() -> nz_iterator
    {
        return nz_iterator(new coo_nz_iterator(*this, m_coords.cbegin(), m_storage.begin()));
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::nz_end() -> nz_iterator
    {
        return nz_iterator(new coo_nz_iterator(*this, m_coords.cend(), m_storage.end()));
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::nz_begin() const -> const_nz_iterator
    {
        return nz_cbegin();
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::nz_end() const -> const_nz_iterator
    {
        return nz_cend();
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::nz_cbegin() const -> const_nz_iterator
    {
        return const_nz_iterator(new coo_const_nz_iterator(*this, m_coords.cbegin(), m_storage.cbegin()));
    }

    template <class P, class C, class ST, class IT>
    inline auto xsparse_coo_scheme<P, C, ST, IT>::nz_cend() const -> const_nz_iterator
    {
        return const_nz_iterator(new coo_const_nz_iterator(*this, m_coords.cend(), m_storage.cend()));
    }
}

#endif

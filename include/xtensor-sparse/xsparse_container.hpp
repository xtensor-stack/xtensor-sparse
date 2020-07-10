#ifndef XSPARSE_XSPARSE_CONTAINER_HPP
#define XSPARSE_XSPARSE_CONTAINER_HPP

#include <xtensor/xaccessible.hpp>
#include <xtensor/xiterable.hpp>
#include <xtensor/xstrides.hpp>

#include "xsparse_reference.hpp"

namespace xt
{
    /*********************
     * xsparse_container *
     *********************/

    template <class D>
    class xsparse_container: public xiterable<D>,
                             private xaccessible<D>
    {
    public:

        using derived_type = D;
        
        using inner_types = xcontainer_inner_types<D>;
        using scheme_type = typename inner_types::scheme_type;
        using index_type = typename inner_types::index_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using bool_load_type = xt::bool_load_type<value_type>;
        using temporary_type = typename inner_types::temporary_type;

        using shape_type = typename inner_types::shape_type;
        using inner_shape_type = typename inner_types::inner_shape_type;
        using strides_type = typename inner_types::strides_type;

        using accessible_base = xaccessible<D>;
        using iterable_base = xiterable<D>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = layout_type::row_major;
        static constexpr bool contiguous_layout = false;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const inner_shape_type& shape() const noexcept;

        template <class S = shape_type>
        void resize(S&& shape, bool force);
        template <class S = shape_type>
        void resize(S&& shape);

        template <class S = shape_type>
        auto& reshape(S&& shape) &;
        template <class T>
        auto& reshape(std::initializer_list<T> shape) &;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        using accessible_base::shape;
        using accessible_base::at;
        using accessible_base::operator[];
        using accessible_base::periodic;
        using accessible_base::in_bounds;

        template <class It>
        reference element(It first, It last);
        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        inline layout_type layout() const noexcept;
        inline bool is_contiguous() const noexcept;

    protected:

        xsparse_container() = default;
        ~xsparse_container() = default;

        xsparse_container(const xsparse_container&) = default;
        xsparse_container& operator=(const xsparse_container&) = default;

        xsparse_container(xsparse_container&&) = default;
        xsparse_container& operator=(xsparse_container&&) = default;

    private:

        template <class S = shape_type>
        void reshape_impl(S&& shape, std::false_type);
        template <class S = shape_type>
        void reshape_impl(S&& shape, std::true_type);

        index_type make_index() const;

        template <class Arg, class... Args>
        index_type make_index(Arg arg, Args... args) const;

        template <class It>
        index_type make_index_from_it(It first, It last) const;

        const_reference access_impl(index_type index) const;
        reference access_impl(index_type index);

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const & noexcept;
        derived_type derived_cast() && noexcept;

        static const value_type ZERO;

        inner_shape_type m_shape;
        strides_type m_strides;
        scheme_type m_scheme;

        friend class xconst_accessible<D>;
        friend class xaccessible<D>;
    };

    /************************************
     * xsparse_container implementation *
     ************************************/

    template <class D>
    const typename xsparse_container<D>::value_type
    xsparse_container<D>::ZERO = 0;

    template <class D>
    inline auto xsparse_container<D>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    template <class D>
    inline auto xsparse_container<D>::dimension() const noexcept -> size_type
    {
        return shape().size();
    }

    template <class D>
    inline auto xsparse_container<D>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class D>
    template <class S>
    inline void xsparse_container<D>::resize(S&& shape, bool force)
    {
        std::size_t dim = shape.size();
        if (m_shape.size() != dim || !std::equal(std::begin(shape), std::end(shape), std::begin(m_shape)) || force)
        {
            m_shape = xtl::forward_sequence<shape_type, S>(shape);
            strides_type old_strides = m_strides;
            resize_container(m_strides, dim);
            compute_strides(m_shape, XTENSOR_DEFAULT_LAYOUT, m_strides);
            m_scheme.update_entries(old_strides, m_strides, m_shape);
        }
    }

    template <class D>
    template <class S>
    inline void xsparse_container<D>::resize(S&& shape)
    {
        resize(std::forward<S>(shape), true);
    }

    template <class D>
    template <class S>
    inline auto& xsparse_container<D>::reshape(S&& shape) &
    {
        reshape_impl(std::forward<S>(shape), std::is_signed<std::decay_t<typename std::decay_t<S>::value_type>>());
        return *this;
    }

    template <class D>
    template <class T>
    inline auto& xsparse_container<D>::reshape(std::initializer_list<T> shape) &
    {
        using sh_type = rebind_container_t<D, shape_type>;
        sh_type sh = xtl::make_sequence<sh_type>(shape.size());
        std::copy(shape.begin(), shape.end(), sh.begin());
        reshape_impl(std::move(sh), std::is_signed<T>());
        return *this;
    }

    template<class D>
    template<class... Args>
    inline auto xsparse_container<D>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(make_index(static_cast<size_type>(args)...));
    }

    template<class D>
    template<class... Args>
    inline auto xsparse_container<D>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return access_impl(make_index(static_cast<size_type>(args)...));
    }


    template <class D>
    template <class It>
    inline auto xsparse_container<D>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return access_impl(make_index_from_it(first, last));
    }

    template <class D>
    template <class It>
    inline auto xsparse_container<D>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return access_impl(make_index_from_it(first, last));
    }

    template <class D>
    template <class S>
    inline bool xsparse_container<D>::broadcast_shape(S& shape, bool) const
    {
        return xt::broadcast_shape(this->shape(), shape);
    }

    template <class D>
    template <class S>
    inline auto xsparse_container<D>::stepper_begin(const S& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(&(this->derived_cast()), offset);
    }

    template <class D>
    template <class S>
    inline auto xsparse_container<D>::stepper_end(const S& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(&(this->derived_cast()), offset, true);
    }

    template <class D>
    template <class S>
    inline auto xsparse_container<D>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(&(this->derived_cast()), offset);
    }

    template <class D>
    template <class S>
    inline auto xsparse_container<D>::stepper_end(const S& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(&(this->derived_cast()), offset, true);
    }

    template <class D>
    inline auto xsparse_container<D>::layout() const noexcept -> layout_type
    {
        return static_layout;
    }

    template <class D>
    inline bool xsparse_container<D>::is_contiguous() const noexcept
    {
        return false;
    }

    template <class D>
    template <class S>
    inline void xsparse_container<D>::reshape_impl(S&& shape, std::false_type /* is unsigned */)
    {
        if (compute_size(shape) != this->size())
        {
            XTENSOR_THROW(std::runtime_error, "Cannot reshape with incorrect number of elements. Do you mean to resize?");
        }

        std::size_t dim = shape.size();
        strides_type old_strides = m_strides;
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        resize_container(m_strides, dim);
        compute_strides(m_shape, XTENSOR_DEFAULT_LAYOUT, m_strides);
        m_scheme.update_entries(old_strides, m_strides, m_shape);
    }

    template <class D>
    template <class S>
    inline void xsparse_container<D>::reshape_impl(S&& _shape, std::true_type /* is signed */)
    {
        using value_type = typename std::decay_t<S>::value_type;
        auto new_size = compute_size(_shape);
        if (this->size() % new_size)
        {
            XTENSOR_THROW(std::runtime_error, "Negative axis size cannot be inferred. Shape mismatch.");
        }
        std::decay_t<S> shape = _shape;
        value_type accumulator = 1;
        std::size_t neg_idx = 0;
        std::size_t i = 0;
        for(auto it = shape.begin(); it != shape.end(); ++it, i++)
        {
            auto&& dim = *it;
            if(dim < 0)
            {
                XTENSOR_ASSERT(dim == -1 && !neg_idx);
                neg_idx = i;
            }
            accumulator *= dim;
        }
        if(accumulator < 0)
        {
            shape[neg_idx] = static_cast<value_type>(this->size()) / std::abs(accumulator);
        }
        else if(this->size() != new_size)
        {
            XTENSOR_THROW(std::runtime_error, "Cannot reshape with incorrect number of elements. Do you mean to resize?");
        }
        
        std::size_t dim = shape.size();
        auto old_strides = m_strides;
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        resize_container(m_strides, dim);
        compute_strides(m_shape, XTENSOR_DEFAULT_LAYOUT, m_strides);
        m_scheme.update_entries(old_strides, m_strides, m_shape);
    }

    template <class D>
    inline auto xsparse_container<D>::make_index() const -> index_type
    {
        return index_type();
    }

    template <class D>
    template <class Arg, class... Args>
    inline auto xsparse_container<D>::make_index(Arg arg, Args... args) const -> index_type
    {
        constexpr size_t argsize = sizeof...(Args) + size_t(1);
        size_type dim = dimension();
        if (argsize == dim)
        {
            return index_type{arg, args...};
        }
        else if(argsize > dim)
        {
            return make_index(args...);
        }
        else
        {
            std::array<Arg, argsize> tmp_index = {arg, args...};
            index_type res = xtl::make_sequence<index_type>(dim);
            std::fill(res.begin(), res.begin() + res.size() - argsize, size_type(0));
            std::copy(tmp_index.cbegin(), tmp_index.cend(), res.begin() + res.size() - argsize);
            return res;
        }
    }

    template <class D>
    template <class It>
    inline auto xsparse_container<D>::make_index_from_it(It first, It last) const -> index_type
    {
        auto index = xtl::make_sequence<index_type>(static_cast<std::size_t>(std::distance(first, last)));
        std::copy(first, last, index.begin());
        return index;
    }

    template<class D>
    inline auto xsparse_container<D>::access_impl(index_type index) const -> const_reference
    {
        auto it = m_scheme.find_element(index);
        if (it)
        {
            return *it;
        }
        return ZERO;
    }

    template<class D>
    inline auto xsparse_container<D>::access_impl(index_type index)-> reference
    {
        auto it = m_scheme.find_element(index);
        value_type v = (it)? *it: value_type();
        return reference(m_scheme, std::move(index), v);
    }

    template <class D>
    inline auto xsparse_container<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xsparse_container<D>::derived_cast() const & noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline auto xsparse_container<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }
}

#endif


#ifndef XSPARSE_XSPARSE_CONTAINER_HPP
#define XSPARSE_XSPARSE_CONTAINER_HPP

#include <tuple>

namespace xt
{
    namespace detail
    {
        template <std::size_t I, class F, class... T, class... D>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        transform_impl(F&& /*f*/, const std::tuple<T...>& /*t*/, std::tuple<D...>& /*d*/) noexcept
        {
        }

        template <std::size_t I, class F, class... T, class... D>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        transform_impl(F&& f, const std::tuple<T...>& t, std::tuple<D...>& d)
            noexcept(noexcept(f(std::get<I>(t))))
        {
            std::get<I>(d) = f(std::get<I>(t));
            for_each_impl<I + 1, F, T..., D...>(std::forward<F>(f), t, d);
        }
    }

    template <class F, class... T, class... D>
    inline void transform(F&& f, const std::tuple<T...>& t, const std::tuple<D...>& dest)
        noexcept(noexcept(detail::transform_impl<0, F, T..., D...>(std::forward<F>(f), t, d)))
    {
        static_assert(sizeof...(T) == sizeof...(D));
        detail::transform_impl<0, F, T..., D...>(std::forward<F>(f), t, d);
    }

    namespace detail
    {
        template <std::size_t I, class... T>
        inline typename std::enable_if<I == sizeof...(T), std::integral_constant<std::size_t>>::type
        min_element_impl(const std::tuple<T...>& /*t*/) noexcept
        {
            return std::integral_constant<std::size_t>(std::numeric_limits<std::size_t>::max());
        }

        template <std::size_t I, class F, class... T, class... D>
        inline typename std::enable_if<I < sizeof...(T), std::integral_constant<std::size_t>>::type
        min_element_impl(const std::tuple<T...>& t) noexcept
        {
            std::integral_constant<std::size_t> index = min_element_impl<I + 1, T...>(t);
            if(std::get<I>(t) != nullptr)
            {
                if (index::value == std::numeric_limits<std::size_t>::max())
                {
                    index = std::integral_constant<std::size_t>(I);
                }
                else
                {
                    if (*std::get<I>(t) < *std::get<index::value>(t))
                    {
                        index = std::integral_constant<std::size_t>(I);
                    }
                }
            }
        }
    }

    template <class F, class... T, class... D>
    inline void min_element_(const std::tuple<T...>& t, const std::tuple<D...>& dest)
        noexcept(noexcept(detail::transform_impl<0, F, T..., D...>(std::forward<F>(f), t, d)))
    {
        static_assert(sizeof...(T) == sizeof...(D));
        detail::transform_impl<0, F, T..., D...>(std::forward<F>(f), t, d);
    }
}

#endif
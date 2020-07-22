#ifndef XSPARSE_UTILS_HPP
#define XSPARSE_UTILS_HPP

#include <tuple>

namespace xt
{
    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        transform_impl(F&& /*f*/, std::tuple<T...>& /*t*/, std::tuple<T*...>& /*d*/) //noexcept
        {
        }

        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        transform_impl(F&& f, std::tuple<T...>& t, std::tuple<T*...>& d)
            noexcept(noexcept(f(std::get<I>(t))))
        {
            std::get<I>(d) = f(std::get<I>(t));
            transform_impl<I + 1, F, T...>(std::forward<F>(f), t, d);
        }
    }

    template <class F, class... T>
    inline void transform(F&& f, std::tuple<T...>& t, std::tuple<T*...>& dest)
        noexcept(noexcept(detail::transform_impl<0, F, T...>(std::forward<F>(f), t, dest)))
    {
        detail::transform_impl<0, F, T...>(std::forward<F>(f), t, dest);
    }
}

#endif
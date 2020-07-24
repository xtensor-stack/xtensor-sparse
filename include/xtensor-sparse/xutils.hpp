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
            noexcept(noexcept(f(I, std::get<I>(t))))
        {
            std::get<I>(d) = f(I, std::get<I>(t));
            transform_impl<I + 1, F, T...>(std::forward<F>(f), t, d);
        }
    }

    template <class F, class... T>
    inline void transform(F&& f, std::tuple<T...>& t, std::tuple<T*...>& dest)
        noexcept(noexcept(detail::transform_impl<0, F, T...>(std::forward<F>(f), t, dest)))
    {
        detail::transform_impl<0, F, T...>(std::forward<F>(f), t, dest);
    }

    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        update_it_impl(F&& /*f*/, std::tuple<T...>& /*t*/, std::tuple<T...>& /*sentinel*/, std::tuple<T*...>& /*d*/) //noexcept
        {
        }

        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        update_it_impl(F&& f, std::tuple<T...>& t, std::tuple<T...>& sentinel, std::tuple<T*...>& d)
            noexcept(noexcept(f(I, std::get<I>(t), std::get<I>(sentinel), std::get<I>(d))))
        {
            f(I, std::get<I>(t), std::get<I>(sentinel), std::get<I>(d));
            update_it_impl<I + 1, F, T...>(std::forward<F>(f), t, sentinel, d);
        }
    }

    template <class F, class... T>
    inline void update_it(F&& f, std::tuple<T...>& t, std::tuple<T...>& sentinel, std::tuple<T*...>& dest)
        noexcept(noexcept(detail::update_it_impl<0, F, T...>(std::forward<F>(f), t, sentinel, dest)))
    {
        detail::update_it_impl<0, F, T...>(std::forward<F>(f), t, sentinel, dest);
    }

    /*****************************
     * accumulate implementation *
     *****************************/

    /// @cond DOXYGEN_INCLUDE_NOEXCEPT

    namespace detail
    {
        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I == sizeof...(T), R>
        accumulate_impl(F&& /*f*/, R init, const std::tuple<T...>& /*t*/, const std::array<bool, sizeof...(T)>& /*is_valid*/) noexcept
        {
            return init;
        }

        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I < sizeof...(T), R>
        accumulate_impl(F&& f, R init, const std::tuple<T...>& t, const std::array<bool, sizeof...(T)>& is_valid)
            noexcept(noexcept(f(init, std::get<I>(t), true)))
        {
            R res = f(init, std::get<I>(t), is_valid[I]);
            return accumulate_impl<I + 1, F, R, T...>(std::forward<F>(f), res, t, is_valid);
        }
    }

    template <class F, class R, class... T>
    inline R accumulate(F&& f, R init, const std::tuple<T...>& t, const std::array<bool, sizeof...(T)>& is_valid)
        noexcept(noexcept(detail::accumulate_impl<0, F, R, T...>(std::forward<F>(f), init, t, is_valid)))
    {
        return detail::accumulate_impl<0, F, R, T...>(std::forward<F>(f), init, t, is_valid);
    }
}

#endif
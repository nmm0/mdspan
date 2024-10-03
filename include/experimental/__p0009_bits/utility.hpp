#pragma once

#include <cstddef>
#include <type_traits>
#include <array>
#include <utility>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// type alias used for rank-based tag dispatch
//
// this is used to enable alternatives to constexpr if when building for C++14
//
template <std::size_t N>
using with_rank = std::integral_constant<std::size_t, N>;

template <class I1, class I2>
MDSPAN_INLINE_FUNCTION
constexpr bool common_integral_compare(I1 x, I2 y)
{
  static_assert(std::is_integral<I1>::value &&
                std::is_integral<I2>::value, "");

  using I = std::common_type_t<I1, I2>;
  return static_cast<I>(x) == static_cast<I>(y);
}

template <class T1, class T2, class F>
MDSPAN_INLINE_FUNCTION
constexpr bool rankwise_equal(with_rank<0>, const T1&, const T2&, F)
{
  return true;
}

template <std::size_t N, class T1, class T2, class F>
MDSPAN_INLINE_FUNCTION
constexpr bool rankwise_equal(with_rank<N>, const T1& x, const T2& y, F func)
{
  bool match = true;

  for (std::size_t r = 0; r < N; r++) {
    match = match && common_integral_compare(func(x, r), func(y, r));
  }

  return match;
}

constexpr struct
{
  template <class T, class I>
  MDSPAN_INLINE_FUNCTION
  constexpr auto operator()(const T& x, I i) const
  {
    return x.extent(i);
  }
} extent;

constexpr struct
{
  template <class T, class I>
  MDSPAN_INLINE_FUNCTION
  constexpr auto operator()(const T& x, I i) const
  {
    return x.stride(i);
  }
} stride;

template<class T>
MDSPAN_INLINE_FUNCTION
constexpr void maybe_unused_variable(const T&) {}


// same as std::integral_constant but with __host__ __device__ annotations on
// the implicit conversion function and the call operator
template <class T, T v>
struct integral_constant {
  using value_type         = T;
  using type               = integral_constant<T, v>;

  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr integral_constant() = default;
  
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr integral_constant(std::integral_constant<T,v>) {};

  static constexpr T value = v;
  MDSPAN_INLINE_FUNCTION constexpr operator value_type() const noexcept {
    return value;
  }
  MDSPAN_INLINE_FUNCTION constexpr value_type operator()() const noexcept {
    return value;
  }
  MDSPAN_INLINE_FUNCTION constexpr operator std::integral_constant<T,v>() const noexcept {
    return std::integral_constant<T,v>{};
  }
};

template<class T, size_t Idx>
struct tuple_member {
  using type = T;
  static constexpr size_t idx = Idx;
  T val;
  MDSPAN_FUNCTION constexpr T& get() { return val; }
  MDSPAN_FUNCTION constexpr const T& get() const { return val; }
};

template<size_t SearchIdx, size_t Idx, class T>
struct tuple_idx_matcher {
  using type = tuple_member<T, Idx>;
  template<class Other>
  MDSPAN_FUNCTION
  constexpr auto operator + (Other v) const {
    if constexpr (Idx == SearchIdx) { return *this; }
    else { return v; }
  }
};

template<class IdxSeq, class ... Elements>
struct tuple_impl;

template<size_t ... Idx, class ... Elements>
struct tuple_impl<std::index_sequence<Idx...>, Elements...>: public tuple_member<Elements, Idx> ... {

  MDSPAN_FUNCTION
  constexpr tuple_impl(Elements ... vals):tuple_member<Elements, Idx>{vals}... {}

  template<size_t N>
  MDSPAN_FUNCTION
  constexpr auto& get() {
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() + ...) );
    return base_t::type::get();
  }
  template<size_t N>
  MDSPAN_FUNCTION
  constexpr const auto& get() const {
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() + ...) );
    return base_t::type::get();
  }
};

// A simple tuple-like class for representing slices internally and is compatible with device code
// This doesn't support type access since we don't need it
// This is not meant as an external API
template<class ... Elements>
struct tuple: public tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...> {
  MDSPAN_FUNCTION
  constexpr tuple(Elements ... vals):tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements ...>(vals ...) {}
};

template<size_t Idx, class ... Args>
MDSPAN_FUNCTION
constexpr auto& get(tuple<Args...>& vals) { return vals.template get<Idx>(); }

template<size_t Idx, class ... Args>
MDSPAN_FUNCTION
constexpr const auto& get(const tuple<Args...>& vals) { return vals.template get<Idx>(); }

template<class ... Elements>
tuple(Elements ...) -> tuple<Elements...>;

template<class T, size_t ... Idx>
constexpr auto c_array_to_std(std::index_sequence<Idx...>, const T(&values)[sizeof...(Idx)]) {
  return std::array{values[Idx]...};
}
template<class T, size_t N>
constexpr auto c_array_to_std(const T(&values)[N]) {
  return c_array_to_std(std::make_index_sequence<N>(), values);
}

} // namespace detail

constexpr struct mdspan_non_standard_tag {
} mdspan_non_standard;

} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

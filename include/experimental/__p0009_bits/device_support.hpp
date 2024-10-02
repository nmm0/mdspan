#pragma once
#include <type_traits>
#include "macros.hpp"
#if !defined(_MDSPAN_HAS_CUDA) && !defined(_MDSPAN_HAS_HIP)
#include <tuple>
#endif

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

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

template<class SearchT, size_t Idx, class T>
struct tuple_type_matcher {
  using type = tuple_member<T, Idx>;
  template<class Other>
  MDSPAN_FUNCTION
  constexpr auto operator + (Other v) const {
    if constexpr (std::is_same_v<T, SearchT>) { return *this; }
    else { return v; }
  }
};

template<class IdxSeq, class ... Elements>
struct tuple_impl;

template<size_t ... Idx, class ... Elements>
struct tuple_impl<std::index_sequence<Idx...>, Elements...>: public tuple_member<Elements, Idx> ... {

  MDSPAN_FUNCTION
  constexpr tuple_impl(Elements ... vals):tuple_member<Elements, Idx>{vals}... {}

  template<class T>
  MDSPAN_FUNCTION
  constexpr T& get() {
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() + ...) );
    return base_t::type::get();
  }
  template<class T>
  MDSPAN_FUNCTION
  constexpr const T& get() const {
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() + ...) );
    return base_t::type::get();
  }

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

template<class ... Elements>
struct tuple: public tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...> {
  MDSPAN_FUNCTION
  constexpr tuple(Elements ... vals):tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements ...>(vals ...) {}
};

template<class T, class ... Args>
MDSPAN_FUNCTION
constexpr auto& get(tuple<Args...>& vals) { return vals.template get<T>(); }

template<class T, class ... Args>
MDSPAN_FUNCTION
constexpr const T& get(const tuple<Args...>& vals) { return vals.template get<T>(); }

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
}
}

//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#pragma once

#include <cassert>
#include "layout_padded_fwd.hpp"
#include "../__p0009_bits/dynamic_extent.hpp"
#include "../__p0009_bits/extents.hpp"
#include "../__p0009_bits/mdspan.hpp"
#include "../__p0009_bits/layout_left.hpp"
#include "../__p0009_bits/layout_right.hpp"
#include "../__p0009_bits/layout_stride.hpp"

namespace std {
namespace experimental {

namespace detail {
template<class _T>
MDSPAN_INLINE_FUNCTION
constexpr auto
__find_aligned_offset(_T __alignment, _T __offset)
{
  if ( __alignment == 0 )
    return _T(0);
  else
    return ( ( __offset + __alignment - 1 ) / __alignment) * __alignment;
}

template<class _ExtentsType, size_t _PaddingStride>
MDSPAN_INLINE_FUNCTION
constexpr size_t
__get_actual_padding_stride()
{
  constexpr auto __rank = _ExtentsType::rank();

  if constexpr (__rank <= size_t(1)) {
    return _PaddingStride;
  } else if constexpr (_PaddingStride != dynamic_extent &&
                       _ExtentsType::static_extent(0) != dynamic_extent) {
    static_assert((_PaddingStride != 0) || (_ExtentsType::static_extent(0) == 0), "padding stride can be 0 only if extents_type::static_extent(0) is 0");
    return __find_aligned_offset(_PaddingStride, _ExtentsType::static_extent(0));
  } else {
    return dynamic_extent;
  }
}

template<size_t _ExtentToSub, class _Extents, size_t _NewExtent, class _Indices>
struct __substitute_extents_impl;

template<size_t _ExtentToSub, class _IndexType, size_t... _OrigExtents, size_t _NewExtent, size_t... _Indices>
struct __substitute_extents_impl<_ExtentToSub, extents<_IndexType, _OrigExtents...>, _NewExtent, index_sequence<_Indices...>>
{
  using __orig_extents_type = extents<_IndexType, _OrigExtents...>;
  using __type = extents<_IndexType, ((_Indices == _ExtentToSub) ? _NewExtent : _OrigExtents)...>;

  template <typename _T>
  static constexpr auto
  __construct_with_type(const __orig_extents_type &__extents, const extents<_IndexType, _NewExtent> &__new_extents)
  {
    return _T{((_Indices == _ExtentToSub) ? __new_extents.extent(0) : __extents.extent(_Indices))...};
  }

  MDSPAN_INLINE_FUNCTION
  static constexpr auto
  __construct(const __orig_extents_type &__extents, const extents<_IndexType, _NewExtent> &__new_extents)
  {
    return __construct_with_type<__type>(__extents, __new_extents);
  }

  MDSPAN_INLINE_FUNCTION
  static constexpr auto
  __construct(const __orig_extents_type &__extents)
  {
    return __type{__extents.extent(_Indices)...};
  }
};

template<size_t _ExtentToSub, class _Extents, size_t _NewExtent>
using __substitute_extents = __substitute_extents_impl<_ExtentToSub, _Extents, _NewExtent, make_index_sequence<_Extents::rank()>>;

template<class _Extents, size_t _ActualPaddingStride, class _Enabled = void>
struct __inner_extents_left
{
  using __subs_type = __substitute_extents< 0, _Extents, _ActualPaddingStride >;
  using __type = typename __subs_type::__type;

  template<size_t _PaddingStride>
  MDSPAN_INLINE_FUNCTION
  static constexpr auto
  __construct(const _Extents &__extents)
  {
    if constexpr (_PaddingStride == dynamic_extent)
    {
      return __subs_type::__construct(__extents);
    } else {
      // This is a corner case where `_PaddingStride` is not dynamic but `_ActualPaddingStride` is
      // because `_Extents::static_extents(0)` is dynamic. We need to initialize it with a run-time
      // value that is the least multiple of `_PaddingStride` greater than or equal to `_Extents::static_extents(0)`
      if constexpr (_Extents::static_extent(0) == dynamic_extent)
      {
        const auto __s_left = __find_aligned_offset(_PaddingStride, __extents.extent(0));
        return __subs_type::__construct(__extents, extents<typename _Extents::index_type, _ActualPaddingStride>(__s_left));
      } else {
        return __subs_type::__construct(__extents, extents<typename _Extents::index_type, _ActualPaddingStride>{});
      }
    }
  }

  template <size_t _PaddingStride, class _Size>
  MDSPAN_INLINE_FUNCTION static constexpr auto
  __construct(const _Extents &__extents, _Size __padding_value)
  {
    const auto __s_left = __find_aligned_offset(__padding_value, __extents.extent(0));

    return __subs_type::__construct(__extents, extents<typename _Extents::index_type, _ActualPaddingStride>(__s_left));
  }

  template <size_t _PaddingStride, class _IndexType>
  MDSPAN_INLINE_FUNCTION static constexpr auto
  __construct_other(const _Extents &__extents, _IndexType __stride0)
  {
    return __subs_type::__construct(__extents, extents<typename _Extents::index_type, _ActualPaddingStride>(__stride0));
  }
};

template<class _Extents, size_t _ActualPaddingStride>
struct __inner_extents_left<_Extents, _ActualPaddingStride, enable_if_t<_Extents::rank() <= size_t(1)>>
{
  using __type = _Extents;

  template<size_t _PaddingStride>
  MDSPAN_INLINE_FUNCTION
  static constexpr const _Extents &
  __construct(const _Extents &__extents)
  {
    return __extents;
  }

  template <size_t _PaddingStride, class _Size>
  MDSPAN_INLINE_FUNCTION static constexpr auto
  __construct(const _Extents &__extents, _Size)
  {
    return __extents;
  }

  template <size_t _PaddingStride, class _IndexType>
  MDSPAN_INLINE_FUNCTION static constexpr auto
  __construct_other(const _Extents &__extents, _IndexType)
  {
    return __extents;
  }
};

template <class _Extents, typename _Enabled = void>
struct __unpadded_extent_type_impl
{
  using __type = extents<typename _Extents::index_type, _Extents::static_extent(0)>;

  MDSPAN_INLINE_FUNCTION
  static constexpr auto
  __construct(const _Extents &__extents)
  {
    return __type(__extents.extent(0));
  }
};

template <class _Extents>
struct __unpadded_extent_type_impl<_Extents, enable_if_t<(_Extents::rank() == 0)>>
{
  using __type = extents<typename _Extents::index_type>;

  MDSPAN_INLINE_FUNCTION
  static constexpr auto
  __construct([[maybe_unused]] const _Extents &__extents)
  {
    return __type{};
  }
};
}

template <size_t padding_stride>
template <class Extents>
class layout_left_padded<padding_stride>::mapping {
public:
  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;
  using layout_type = layout_left_padded<padding_stride>;

#ifndef MDSPAN_INTERNAL_TEST
private:
#endif // MDSPAN_INTERNAL_TEST

  static_assert((padding_stride != 0) || (extents_type::static_extent(0) == 0) || (extents_type::static_extent(0) == dynamic_extent),
                "if padding stride is 0, static_extent(0) must also be 0 or dynamic_extent");

  static constexpr size_t __actual_padding_stride = detail::__get_actual_padding_stride<extents_type, padding_stride>();

  using __inner_extents_type = typename detail::__inner_extents_left<extents_type, __actual_padding_stride>::__type;
  using __unpadded_extent_type = typename detail::__unpadded_extent_type_impl<extents_type>::__type;
  using __inner_mapping_type = layout_left::template mapping<__inner_extents_type>;

  __inner_mapping_type __inner_mapping;
  __unpadded_extent_type __unpadded_extent;

public:
#if !MDSPAN_HAS_CXX_20
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr mapping()
      : mapping(extents_type{})
  {}
#else
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr mapping()
    requires(__actual_padding_stride != dynamic_extent) = default;

  MDSPAN_INLINE_FUNCTION
  constexpr mapping()
    requires(__actual_padding_stride == dynamic_extent)
      : mapping(extents_type{})
  {}
#endif

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(const mapping&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED mapping& operator=(const mapping&) noexcept = default;

  /**
   * Initializes the mapping with the given extents.
   *
   * \param ext the given extents
   */
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const extents_type& __ext)
    : __inner_mapping(detail::__inner_extents_left<extents_type, __actual_padding_stride>::template __construct<padding_stride>(__ext)),
      __unpadded_extent(detail::__unpadded_extent_type_impl<extents_type>::__construct(__ext))
  {}

  /**
   * Initializes the mapping with the given extents and the specified padding value.
   *
   * This overload participates in overload resolution only if `is_convertible_v<Size, index_type>`
   * is `true` and `is_nothrow_constructible_v<index_type, Size>` is `true`
   *
   * \param ext the given extents
   * \param padding_value the padding value
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Size,
    /* requires */ (
      is_convertible_v<_Size, index_type>
      && is_nothrow_constructible_v<index_type, _Size>
    )
  )
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const extents_type &__ext, _Size __padding_value)
      : __inner_mapping(detail::__inner_extents_left<extents_type, __actual_padding_stride>::template __construct<padding_stride>(__ext, static_cast<index_type>(__padding_value))),
        __unpadded_extent(detail::__unpadded_extent_type_impl<extents_type>::__construct(__ext))
  {
    assert((padding_stride == dynamic_extent) || (padding_stride == static_cast<index_type>(__padding_value)));
  }

  /**
   * Converting constructor from `layout_left::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true.
   * If `OtherExtents::rank() > 1` then one of `padding_stride`, `static_extent(0)`, or `OtherExtents::static_extent(0)` must be `dynamic_extent`;
   * otherwise, `OtherExtents::static_extent(0)` must be equal to the least multiple of `padding_stride` greater than or equal to `extents_type::static_extent(0)`
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _OtherExtents,
    /* requires */ (
      is_constructible_v<extents_type, _OtherExtents>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((!is_convertible_v<_OtherExtents, extents_type>))
  constexpr mapping(const layout_left::mapping<_OtherExtents> &__other_mapping)
      : __inner_mapping(detail::__inner_extents_left<extents_type, __actual_padding_stride>::template __construct_other<padding_stride>(__other_mapping.extents(), __other_mapping.stride(1))),
        __unpadded_extent(detail::__unpadded_extent_type_impl<extents_type>::__construct(__other_mapping.extents()))
  {
    static_assert(!((_OtherExtents::rank() > 1) && (__actual_padding_stride != dynamic_extent) && (_OtherExtents::static_extent(0) != dynamic_extent))
                  || (__actual_padding_stride == _OtherExtents::static_extent(0)));
  }

  /**
   * Converting constructor from `layout_stride::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _OtherExtents,
    /* requires */ (
      is_constructible_v<extents_type, _OtherExtents>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
  constexpr mapping(const layout_stride::mapping<_OtherExtents> &__other_mapping)
      : __inner_mapping(detail::__inner_extents_left<extents_type, __actual_padding_stride>::template __construct_other<padding_stride>(__other_mapping.extents(), __other_mapping.stride(1))),
        __unpadded_extent(detail::__unpadded_extent_type_impl<extents_type>::__construct(__other_mapping.extents()))
  {
  }

  /**
   * Converting constructor from `layout_left_padded::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true.
   * Either `padding_stride` or `OtherPaddingStride` must be `std::dynamic_extent`, or `padding_stride == OtherPaddingStride`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    size_t _OtherPaddingStride, class _OtherExtents,
    /* requires */ (
      is_constructible_v<extents_type, _OtherExtents>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 1 && (padding_stride == dynamic_extent || _OtherPaddingStride == dynamic_extent)))
  constexpr
  mapping(const typename layout_left_padded<_OtherPaddingStride>::template mapping<_OtherExtents> &__other_mapping)
      : __inner_mapping(detail::__inner_extents_left<extents_type, __actual_padding_stride>::template __construct_other<padding_stride>(__other_mapping.extents(), __other_mapping.stride(1))),
        __unpadded_extent(detail::__unpadded_extent_type_impl<extents_type>::__construct(__other_mapping.extents()))
  {
    static_assert(padding_stride == dynamic_extent || _OtherPaddingStride == dynamic_extent || padding_stride == _OtherPaddingStride);
  }

  /**
   * Converting constructor from `layout_right_padded::mapping`.
   *
   * This overload participates in overload resolution only if `extents_type::rank()` is 0 or 1 and `is_constructible_v<extents_type, OtherExtents>` is `true`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    size_t _OtherPaddingStride, class _OtherExtents,
    /* requires */ (
      extents_type::rank() <= 1
      && is_constructible_v<extents_type, _OtherExtents>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((!is_convertible_v<_OtherExtents, extents_type>))
  constexpr
  mapping(const typename layout_right_padded<_OtherPaddingStride>::template mapping<_OtherExtents> &__other_mapping) noexcept
      : __inner_mapping(__other_mapping.extents()),
        __unpadded_extent(detail::__unpadded_extent_type_impl<extents_type>::__construct(__other_mapping.extents()))
  {}

  constexpr extents_type extents() const noexcept
  {
    if constexpr (extents_type::rank() == size_t(0))
    {
      return {};
    } else {
      return detail::__substitute_extents<0, __inner_extents_type, __unpadded_extent_type::static_extent(0)>::template __construct_with_type<extents_type>(__inner_mapping.extents(), __unpadded_extent);
    }
  }

  constexpr std::array<index_type, extents_type::rank()>
  strides() const noexcept
  {
    array<index_type, extents_type::rank()> __s{};
    for (rank_type __r = 0; __r < extents_type::rank(); ++__r)
    {
      __s[__r] = __inner_mapping.stride(__r);
    }
    return __s;
  }

  constexpr index_type
  required_span_size() const noexcept
  {
    return __inner_mapping.required_span_size();
  }

  /**
   * Return the mapping given the provided indices per rank.
   *
   * This overload participates in overload resolution only if:
   * - `sizeof...(Indices) == extents_type::rank()`,
   * - `(is_convertible_v<Indices, index_type> && ...) is true`, and
   * - (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class... _Indices,
    /* requires */ (
      sizeof...(_Indices) == extents_type::rank()
      && (is_convertible_v<_Indices, index_type> && ...)
      && (is_nothrow_constructible_v<index_type, _Indices> && ...)
    )
  )
  constexpr size_t operator()(_Indices... __idxs) const noexcept
  {
    return __inner_mapping(std::forward<_Indices>(__idxs)...);
  }

  static constexpr bool is_always_unique() noexcept { return true; }
  static constexpr bool is_always_exhaustive() noexcept
  {
    return (extents_type::rank() <= size_t(1))
      || (extents_type::static_extent(0) != dynamic_extent
          && extents_type::static_extent(0) == __inner_extents_type::static_extent(0));
  }
  static constexpr bool is_always_strided() noexcept { return true; }

  static constexpr bool is_unique() noexcept { return true; }
  constexpr bool is_exhaustive() const noexcept
  {
    return (extents_type::rank() == 0)
      || (__inner_mapping.extents().extent(0) == __unpadded_extent.extent(0));
  }
  static constexpr bool is_strided() noexcept { return true; }

  constexpr index_type stride(rank_type __r) const noexcept
  {
    return __inner_mapping.stride(__r);
  }

  /**
   * Equality operator between `layout_left_padded`s
   *
   * This overload only participates in overload resolution if `OtherExtents::rank() == extents_type::rank()`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (
      detail::__is_layout_left_padded_mapping_of<_Mapping>
          && (_Mapping::extents_type::rank() == extents_type::rank())
    )
  )
  friend constexpr bool operator==(const mapping &__left, const _Mapping &__right) noexcept
  {
    return (__left.extents() == __right.extents()) && (!(extents_type::rank() > size_t(1)) || (__left.stride(1) == __right.stride(1)));
  }

  /**
   * Inequality operator between `layout_left_padded`s
   *
   * This overload only participates in overload resolution if `OtherExtents::rank() == extents_type::rank()`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (
      detail::__is_layout_left_padded_mapping_of<_Mapping>
      && (_Mapping::extents_type::rank() == extents_type::rank())
    )
  )
  friend constexpr bool operator!=(const mapping &__left, const _Mapping &__right) noexcept
  {
    return !(__left == __right);
  }
};
}
}
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

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {

namespace detail {
template<class _T>
MDSPAN_INLINE_FUNCTION
constexpr _T
__find_aligned_offset(_T __alignment, _T __offset)
{
  if ( __alignment == 0 ) {
    return _T(0);
  } else {
    return ( ( __offset + __alignment - 1 ) / __alignment) * __alignment;
  }
}

template<class _ExtentsType, size_t _PaddingStride, size_t _ExtentToPadIdx>
MDSPAN_INLINE_FUNCTION
constexpr size_t
__get_actual_static_padding_stride()
{
  constexpr auto __rank = _ExtentsType::rank();

  if constexpr (__rank <= size_t(1)) {
    return _PaddingStride;
  } else if constexpr (_PaddingStride != dynamic_extent &&
                       _ExtentsType::static_extent(_ExtentToPadIdx) != dynamic_extent) {
    static_assert((_PaddingStride != 0) || (_ExtentsType::static_extent(_ExtentToPadIdx) == 0), "padding stride can be 0 only if extents_type::static_extent(extent-to-pad) is 0 or dynamic_extent");
    return __find_aligned_offset(_PaddingStride, _ExtentsType::static_extent(_ExtentToPadIdx));
  } else {
    return dynamic_extent;
  }
}

template <size_t _PaddingStride, typename _Extents, size_t _ExtentToPadIdx, size_t _Rank>
struct __static_array_type_for_padded_extent
{
  static constexpr size_t __padding_stride = _PaddingStride;
  using __index_type = typename _Extents::index_type;
  using __extents_type = _Extents;
  using __type =
      ::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::maybe_static_array<
          __index_type, size_t, dynamic_extent,
          detail::__get_actual_static_padding_stride<
              __extents_type, __padding_stride, _ExtentToPadIdx>()>;
};

template <size_t _PaddingStride, typename _Extents, size_t _ExtentToPadIdx>
struct __static_array_type_for_padded_extent<_PaddingStride, _Extents, _ExtentToPadIdx, 0>
{
  static constexpr size_t __padding_stride = _PaddingStride;
  using __index_type = typename _Extents::index_type;
  using __extents_type = _Extents;
  using __type =
      ::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::maybe_static_array<
          __index_type, size_t, dynamic_extent, 1>;
};

template <size_t _PaddingStride, typename _Extents, size_t _ExtentToPadIdx>
struct __static_array_type_for_padded_extent<_PaddingStride, _Extents, _ExtentToPadIdx, 1>
{
  static constexpr size_t __padding_stride = _PaddingStride;
  using __index_type = typename _Extents::index_type;
  using __extents_type = _Extents;
  using __type =
      ::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::maybe_static_array<
          __index_type, size_t, dynamic_extent, _Extents::static_extent(_ExtentToPadIdx)>;
};

template <size_t _PaddingStride, typename _Extents, size_t _ExtentToPadIdx>
class __padded_extent
{
public:

  static constexpr size_t __padding_stride = _PaddingStride;
  using __index_type = typename _Extents::index_type;
  using __extents_type = _Extents;
  using __static_array_type = typename __static_array_type_for_padded_extent<_PaddingStride, _Extents, _ExtentToPadIdx, _Extents::rank()>::__type;

  MDSPAN_INLINE_FUNCTION constexpr __padded_extent() = default;

  MDSPAN_INLINE_FUNCTION
  constexpr __padded_extent(const _Extents &__extents)
      : m_padding{__impl_init_padding(__extents)} {}

  MDSPAN_INLINE_FUNCTION
  constexpr __padded_extent(const _Extents &__extents,
                            __index_type __padding_value)
      : m_padding(__impl_init_padding(__extents, __padding_value)) {}

  template <typename _Mapping, size_t _PaddingStrideIdx>
  MDSPAN_INLINE_FUNCTION
  constexpr __padded_extent(const _Mapping &__other_mapping, std::integral_constant<size_t, _PaddingStrideIdx> __padding_stride_idx)
    : m_padding(__impl_init_padding(__other_mapping, __padding_stride_idx)) {}

  MDSPAN_INLINE_FUNCTION
  constexpr static size_t static_value() { return __static_array_type::static_value(0); }

  MDSPAN_INLINE_FUNCTION
  constexpr __index_type value() const noexcept { return m_padding.value(0); }

private:

  MDSPAN_INLINE_FUNCTION
  explicit constexpr __padded_extent(__index_type __padding_value)
      : m_padding{__padding_value} {
  }

  MDSPAN_INLINE_FUNCTION
  static constexpr __static_array_type
  __impl_init_padding(const _Extents &__extents) {
    if constexpr (_PaddingStride == dynamic_extent) {
      return {__extents.extent(_ExtentToPadIdx)};
    } else {
      return __impl_init_padding(__extents, __padding_stride);
    }
  }

  MDSPAN_INLINE_FUNCTION static constexpr __static_array_type
      __impl_init_padding(const _Extents &__extents,
                          __index_type __padding_value) {
    if constexpr (_Extents::rank() > 1) {
      return {__find_aligned_offset(
          __padding_value, __extents.extent(_ExtentToPadIdx))};
    } else if constexpr (_Extents::rank() == 1) {
      return {__extents.extent(_ExtentToPadIdx)};
    } else {
      return {};
    }
  }

  template <typename _Mapping, size_t _PaddingStrideIdx>
  MDSPAN_INLINE_FUNCTION static constexpr __static_array_type
  __impl_init_padding(const _Mapping &__other_mapping,
                      std::integral_constant<size_t, _PaddingStrideIdx>) {
    if constexpr (_Extents::rank() > 1) {
      return __static_array_type{__other_mapping.stride(_PaddingStrideIdx)};
    } else if constexpr (_Extents::rank() > 0) {
      return __static_array_type{__other_mapping.extents().extent(_ExtentToPadIdx)};
    } else {
      return __static_array_type{};
    }
  }

  __static_array_type m_padding;
};
}

template <size_t PaddingStride>
template <class Extents>
class layout_left_padded<PaddingStride>::mapping {
public:
  using extents_type = Extents;
  static constexpr size_t padding_stride = PaddingStride;

  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;
  using layout_type = layout_left_padded<padding_stride>;

#ifndef MDSPAN_INTERNAL_TEST
private:
#endif // MDSPAN_INTERNAL_TEST

  static constexpr rank_type __padding_stride_idx = detail::__layout_padded_constants<layout_type, extents_type>::__padding_stride_idx;
  static constexpr rank_type __extent_to_pad_idx = detail::__layout_padded_constants<layout_type, extents_type>::__extent_to_pad_idx;

  static_assert((padding_stride != 0)
                || (extents_type::static_extent(__extent_to_pad_idx) == 0)
                || (extents_type::static_extent(__extent_to_pad_idx) == dynamic_extent),
                "if padding stride is 0, static_extent(0) must also be 0 or dynamic_extent");

  static constexpr size_t __actual_padding_stride = detail::__get_actual_static_padding_stride<extents_type, padding_stride, __extent_to_pad_idx>();

  using __padded_stride_type = detail::__padded_extent< padding_stride, extents_type, __extent_to_pad_idx >;

  __padded_stride_type __padded_stride = {};
  extents_type __extents = {};

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
    : __padded_stride(__ext), __extents(__ext)
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
      std::is_convertible_v<_Size, index_type>
      && std::is_nothrow_constructible_v<index_type, _Size>
    )
  )
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const extents_type &__ext, _Size __padding_value)
      : __padded_stride(__ext, __padding_value), __extents(__ext)
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
      std::is_constructible_v<extents_type, _OtherExtents>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible_v<_OtherExtents, extents_type>))
  constexpr mapping(const layout_left::mapping<_OtherExtents> &__other_mapping)
      : __padded_stride(__other_mapping, std::integral_constant<size_t, __padding_stride_idx>{}),
        __extents(__other_mapping.extents())
  {
    static_assert(!((_OtherExtents::rank() > 1) && (__actual_padding_stride != dynamic_extent) && (_OtherExtents::static_extent(__extent_to_pad_idx) != dynamic_extent))
                  || (__actual_padding_stride == _OtherExtents::static_extent(__extent_to_pad_idx)));
  }

  /**
   * Converting constructor from `layout_stride::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _OtherExtents,
    /* requires */ (
      std::is_constructible_v<extents_type, _OtherExtents>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
  constexpr mapping(const layout_stride::mapping<_OtherExtents> &__other_mapping)
      : __padded_stride(__other_mapping, std::integral_constant<size_t, __padding_stride_idx>{}),
        __extents(__other_mapping.extents())
  {
  }

  /**
   * Converting constructor from `layout_left_padded::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true.
   * Either `padding_stride` or `OtherPaddingStride` must be `std::dynamic_extent`, or `padding_stride == OtherPaddingStride`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (
      detail::__is_layout_left_padded_mapping<_Mapping>::value
      && std::is_constructible_v<extents_type, typename _Mapping::extents_type>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 1 && (padding_stride == dynamic_extent || _Mapping::padding_stride == dynamic_extent)))
  constexpr
  mapping(const _Mapping &__other_mapping)
      : __padded_stride(__other_mapping, std::integral_constant<size_t, __padding_stride_idx>{}),
        __extents(__other_mapping.extents())
  {
    static_assert(padding_stride == dynamic_extent ||
                  _Mapping::padding_stride == dynamic_extent ||
                  padding_stride == _Mapping::padding_stride);
  }

  /**
   * Converting constructor from `layout_right_padded::mapping`.
   *
   * This overload participates in overload resolution only if `extents_type::rank()` is 0 or 1 and `is_constructible_v<extents_type, OtherExtents>` is `true`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (
      detail::__is_layout_right_padded_mapping<_Mapping>::value
      && extents_type::rank() <= 1
      && std::is_constructible_v<extents_type, typename _Mapping::extents_type>
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible_v<typename _Mapping::extents_type, extents_type>))
  constexpr
  mapping(const _Mapping &__other_mapping) noexcept
      : __padded_stride(__other_mapping.extents(), __other_mapping.extents().extent(__extent_to_pad_idx)),
        __extents(__other_mapping.extents())
  {}

  constexpr const extents_type &extents() const noexcept
  {
    return __extents;
  }

  constexpr std::array<index_type, extents_type::rank()>
  strides() const noexcept
  {
    if constexpr ( extents_type::rank() == 0 ) {
      return {};
    } else {
      index_type value = 1;
      std::array<index_type, extents_type::rank()> __s{};
      for (rank_type __r = 0; __r < __extent_to_pad_idx; ++__r)
      {
        __s[__r] = value;
        value *= __extents.extent(__r);
      }
      __s[__extent_to_pad_idx] = value;
      value *= __padded_stride.value();
      if constexpr (__extent_to_pad_idx < extents_type::rank() - 1) {
        for (rank_type __r = __extent_to_pad_idx + 1; __r < extents_type::rank() - 1; ++__r)
        {
          __s[__r] = value;
          value *= __extents.extent(__r);
        }
        __s[extents_type::rank() - 1] = value;
      }
      return __s;
    }
  }

  constexpr index_type
  required_span_size() const noexcept
  {
    if constexpr ( extents_type::rank() == 0 ) {
      return 1;
    } else {
      index_type value = 1;
      for (rank_type __r = 0; __r < __extent_to_pad_idx; ++__r)
      {
        value *= __extents.extent(__r);
      }
      value *= __padded_stride.value();
      for (rank_type __r = __extent_to_pad_idx + 1; __r < extents_type::rank(); ++__r)
      {
        value *= __extents.extent(__r);
      }
      return value;
    }
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
      && (std::is_convertible_v<_Indices, index_type> && ...)
      && (std::is_nothrow_constructible_v<index_type, _Indices> && ...)
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
      || (extents_type::static_extent(__extent_to_pad_idx) != dynamic_extent
          && extents_type::static_extent(__extent_to_pad_idx) == __padded_stride_type::static_value());
  }
  static constexpr bool is_always_strided() noexcept { return true; }

  static constexpr bool is_unique() noexcept { return true; }
  constexpr bool is_exhaustive() const noexcept
  {
    return (extents_type::rank() == 0)
           || (__extents.extent(__extent_to_pad_idx) == __padded_stride.value());
  }
  static constexpr bool is_strided() noexcept { return true; }

  constexpr index_type stride(rank_type __r) const noexcept
  {
    index_type value = 1;
    for (rank_type __i = 0; (__i < __extent_to_pad_idx) && (__i < __r); ++__i)
    {
      value *= __extents.extent(__i);
    }
    if (__extent_to_pad_idx < __r)
      value *= __padded_stride.value();
    for (rank_type __i = __extent_to_pad_idx + 1; __i < __r; ++__i)
    {
      value *= __extents.extent(__i);
    }
    return value;
  }

  /**
   * Equality operator between `layout_left_padded`s
   *
   * This overload only participates in overload resolution if `OtherExtents::rank() == extents_type::rank()`.
   *
   * \note There is currently a difference from p2642r2, where this function is specified as taking
   * `layout_left_padded< padding_stride >::mapping< Extents>`. However, this makes `padding_stride` non-deducible.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (
      detail::__is_layout_left_padded_mapping<_Mapping>::value
      && (_Mapping::extents_type::rank() == extents_type::rank())
    )
  )
  friend constexpr bool operator==(const mapping &__left, const _Mapping &__right) noexcept
  {
    // Workaround for some compilers not short-circuiting properly with compile-time checks
    // i.e. we can't access stride(_padding_stride_idx) of a rank 0 mapping
    bool strides_equal = true;
    if constexpr (extents_type::rank() > size_t(1))
    {
      strides_equal = __left.stride(__padding_stride_idx) == __right.stride(__padding_stride_idx);
    }
    return (__left.extents() == __right.extents()) && strides_equal;
  }

#if !MDSPAN_HAS_CXX_20
  /**
   * Inequality operator between `layout_left_padded`s
   *
   * This overload only participates in overload resolution if `OtherExtents::rank() == extents_type::rank()`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (
      detail::__is_layout_left_padded_mapping<_Mapping>::value
      && (_Mapping::extents_type::rank() == extents_type::rank())
    )
  )
  friend constexpr bool operator!=(const mapping &__left, const _Mapping &__right) noexcept
  {
    return !(__left == __right);
  }
#endif
};

template <size_t PaddingStride>
template <class Extents>
class layout_right_padded<PaddingStride>::mapping {
public:
  using extents_type = Extents;
  static constexpr size_t padding_stride = PaddingStride;

  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;
  using layout_type = layout_right_padded<padding_stride>;

#ifndef MDSPAN_INTERNAL_TEST
  private:
#endif // MDSPAN_INTERNAL_TEST

  static constexpr rank_type __padding_stride_idx = detail::__layout_padded_constants<layout_type, extents_type>::__padding_stride_idx;
  static constexpr rank_type __extent_to_pad_idx = detail::__layout_padded_constants<layout_type, extents_type>::__extent_to_pad_idx;

  static_assert((padding_stride != 0)
                || (extents_type::static_extent(__extent_to_pad_idx) == 0)
                || (extents_type::static_extent(__extent_to_pad_idx) == dynamic_extent),
                "if padding stride is 0, static_extent(extent-to-pad-rank) must also be 0 or dynamic_extent");

  static constexpr size_t __actual_padding_stride = detail::__get_actual_static_padding_stride<extents_type, padding_stride, __extent_to_pad_idx>();

  using __padded_stride_type = detail::__padded_extent< padding_stride, extents_type, __extent_to_pad_idx >;

  __padded_stride_type __padded_stride = {};
  extents_type __extents = {};

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
  constexpr mapping(const extents_type &__ext)
      : __padded_stride(__ext), __extents(__ext) {}

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
          std::is_convertible_v<_Size, index_type>
              && std::is_nothrow_constructible_v<index_type, _Size>
          )
      )
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const extents_type &__ext, _Size __padding_value)
      : __padded_stride(__ext, static_cast<index_type>(__padding_value)),
        __extents(__ext) {
    assert((padding_stride == dynamic_extent) ||
           (padding_stride == static_cast<index_type>(__padding_value)));
  }

  /**
   * Converting constructor from `layout_right::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true.
   * If `OtherExtents::rank() > 1` then one of `padding_stride`, `static_extent(0)`, or `OtherExtents::static_extent(0)` must be `dynamic_extent`;
   * otherwise, `OtherExtents::static_extent(0)` must be equal to the least multiple of `padding_stride` greater than or equal to `extents_type::static_extent(0)`
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (
          std::is_constructible_v<extents_type, _OtherExtents>
          )
      )
  MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible_v<_OtherExtents, extents_type>))
  constexpr mapping(const layout_right::mapping<_OtherExtents> &__other_mapping)
      : __padded_stride(__other_mapping, std::integral_constant<size_t, __padding_stride_idx>{}),
        __extents(__other_mapping.extents())
  {
    static_assert(!((_OtherExtents::rank() > 1) && (__padded_stride_type::static_value() != dynamic_extent) && (_OtherExtents::static_extent(__extent_to_pad_idx) != dynamic_extent))
                  || (__padded_stride_type::static_value() == _OtherExtents::static_extent(__extent_to_pad_idx)));
  }

  /**
   * Converting constructor from `layout_stride::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (
          std::is_constructible_v<extents_type, _OtherExtents>
          )
      )
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
  constexpr mapping(const layout_stride::mapping<_OtherExtents> &__other_mapping)
      : __padded_stride(__other_mapping, std::integral_constant<size_t, __padding_stride_idx>{}),
        __extents(__other_mapping.extents())
  {}

  /**
   * Converting constructor from `layout_right_padded::mapping`.
   *
   * This overload participates in overload resolution only if `is_constructible_v<extents_type, OtherExtents>` is true.
   * Either `padding_stride` or `OtherPaddingStride` must be `std::dynamic_extent`, or `padding_stride == OtherPaddingStride`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (
          detail::__is_layout_right_padded_mapping<_Mapping>::value
              && std::is_constructible_v<extents_type, typename _Mapping::extents_type>
          )
      )
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 1 &&
                               (padding_stride == dynamic_extent ||
                                _Mapping::padding_stride == dynamic_extent)))
  constexpr mapping(const _Mapping &__other_mapping)
      : __padded_stride(__other_mapping, std::integral_constant<size_t, __padding_stride_idx>{}),
        __extents(__other_mapping.extents())
  {
    static_assert(padding_stride == dynamic_extent ||
                  _Mapping::padding_stride == dynamic_extent ||
                  padding_stride == _Mapping::padding_stride);
  }

  /**
   * Converting constructor from `layout_left_padded::mapping`.
   *
   * This overload participates in overload resolution only if `extents_type::rank()` is 0 or 1 and `is_constructible_v<extents_type, OtherExtents>` is `true`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (
          detail::__is_layout_left_padded_mapping<_Mapping>::value
                  && extents_type::rank() <= 1
          && std::is_constructible_v<extents_type, typename _Mapping::extents_type>
          )
      )
  MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible_v<typename _Mapping::extents_type, extents_type>))
  constexpr mapping(const _Mapping &__other_mapping) noexcept
      : __padded_stride(__other_mapping.extents(), __other_mapping.extents().extent(__extent_to_pad_idx)),
        __extents(__other_mapping.extents())
  {}

  constexpr const extents_type &extents() const noexcept
  {
    return __extents;
  }

  constexpr std::array<index_type, extents_type::rank()>
  strides() const noexcept
  {
    if constexpr ( extents_type::rank() == 0 ) {
      return {};
    } else {
      index_type value = 1;
      std::array<index_type, extents_type::rank()> __s{};
      for (rank_type __r = extents_type::rank() - 1; __r > __extent_to_pad_idx; --__r)
      {
        __s[__r] = value;
        value *= __extents.extent(__r);
      }
      __s[__extent_to_pad_idx] = value;
      value *= __padded_stride.value();
      if constexpr ( __extent_to_pad_idx > 0) {
        for (rank_type __r = __extent_to_pad_idx - 1; __r > 0; --__r)
        {
          __s[__r] = value;
          value *= __extents.extent(__r);
        }
        __s[0] = value;
      }
      return __s;
    }
  }

  constexpr index_type
  required_span_size() const noexcept
  {
    if constexpr ( extents_type::rank() == 0 ) {
      return 1;
    } else {
      index_type value = 1;
      for (rank_type __r = 0; __r < __extent_to_pad_idx; ++__r)
      {
        value *= __extents.extent(__r);
      }
      value *= __padded_stride.value();
      for (rank_type __r = __extent_to_pad_idx + 1; __r < extents_type::rank(); ++__r)
      {
        value *= __extents.extent(__r);
      }
      return value;
    }
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
          && (std::is_convertible_v<_Indices, index_type> && ...)
          && (std::is_nothrow_constructible_v<index_type, _Indices> && ...)
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
           || (extents_type::static_extent(__extent_to_pad_idx) != dynamic_extent
               && extents_type::static_extent(__extent_to_pad_idx) == __padded_stride_type::static_value());
  }
  static constexpr bool is_always_strided() noexcept { return true; }

  static constexpr bool is_unique() noexcept { return true; }
  constexpr bool is_exhaustive() const noexcept
  {
    return (extents_type::rank() == 0)
           || (__extents.extent(__extent_to_pad_idx) == __padded_stride.value());
  }
  static constexpr bool is_strided() noexcept { return true; }

  constexpr index_type stride(rank_type __r) const noexcept
  {
    index_type value = 1;
    for (rank_type __i = extents_type::rank() - 1; (__i > __extent_to_pad_idx) && (__i > __r); --__i)
    {
      value *= __extents.extent(__i);
    }
    if (__extent_to_pad_idx > __r)
      value *= __padded_stride.value();
    for (rank_type __i = __extent_to_pad_idx - 1; __i > __r; --__i)
    {
      value *= __extents.extent(__i);
    }
    return value;
  }

  /**
   * Equality operator between `layout_right_padded`s
   *
   * This overload only participates in overload resolution if `OtherExtents::rank() == extents_type::rank()`.
   *
   * \note There is currently a difference from p2642r2, where this function is specified as taking
   * `layout_right_padded< padding_stride >::mapping< Extents>`. However, this makes `padding_stride` non-deducible.
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (
          detail::__is_layout_right_padded_mapping<_Mapping>::value
          && (_Mapping::extents_type::rank() == extents_type::rank())
          )
      )
  friend constexpr bool operator==(const mapping &__left, const _Mapping &__right) noexcept
  {
    // Workaround for some compilers not short-circuiting properly with compile-time checks
    // i.e. we can't access stride(_padding_stride_idx) of a rank 0 mapping
    bool strides_equal = true;
    if constexpr (extents_type::rank() > size_t(1))
    {
      strides_equal = __left.stride(__padding_stride_idx) == __right.stride(__padding_stride_idx);
    }
    return (__left.extents() == __right.extents()) && strides_equal;
  }

#if !MDSPAN_HAS_CXX_20
  /**
   * Inequality operator between `layout_right_padded`s
   *
   * This overload only participates in overload resolution if `OtherExtents::rank() == extents_type::rank()`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (
          detail::__is_layout_right_padded_mapping<_Mapping>::value
          && (_Mapping::extents_type::rank() == extents_type::rank())
          )
      )
  friend constexpr bool operator!=(const mapping &__left, const _Mapping &__right) noexcept
  {
    return !(__left == __right);
  }
#endif
};
}
}

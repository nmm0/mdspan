/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/


#pragma once

#include "../mdspan"
#include <cassert>

namespace std {
namespace experimental {

namespace {
  template<class Extents>
  struct size_of_extents;

  template<size_t ... Extents>
  struct size_of_extents<extents<Extents...>> {
    constexpr static size_t value() {
      size_t size = 1;
      for(int r=0; r<extents<Extents...>::rank(); r++) size *= extents<Extents...>::static_extent(r);
      return size;
    }
  };
}

template <
  class ElementType,
  class Extents,
  class LayoutPolicy = layout_right,
  class Container = conditional_t<(Extents::rank_dynamic()>0), vector<ElementType>, array<ElementType, size_of_extents<Extents>::value()>>
>
class mdarray {
private:
  static_assert(detail::__is_extents_v<Extents>, "std::experimental::mdspan's Extents template parameter must be a specialization of std::experimental::extents.");


public:

  //--------------------------------------------------------------------------------
  // Domain and codomain types

  using extents_type = Extents;
  using layout_type = LayoutPolicy;
  using container_type = Container;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;
  using value_type = remove_cv_t<element_type>;
  using size_type = typename Extents::size_type;
  using pointer = typename container_type::pointer;
  using reference = typename container_type::reference;
  using const_pointer = typename container_type::const_pointer;
  using const_reference = typename container_type::const_reference;

public:

  //--------------------------------------------------------------------------------
  // [mdspan.basic.cons], mdspan constructors, assignment, and destructor

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdarray() requires(extents_type::rank_dynamic()!=0) = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdarray(const mdarray&) = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdarray(mdarray&&) = default;

  // Constructors for container types constructible from a size
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      _MDSPAN_TRAIT(is_constructible, extents_type, SizeTypes...) &&
      _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type) &&
      _MDSPAN_TRAIT(is_constructible, container_type, size_t)
    )
  )
  MDSPAN_INLINE_FUNCTION
  explicit constexpr mdarray(SizeTypes... dynamic_extents)
    : map_(extents_type(dynamic_extents...)), ctr_(map_.required_span_size())
  { }

  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdarray, (const extents_type& exts), ,
    /* requires */ (_MDSPAN_TRAIT(is_constructible, container_type, size_t) &&
                    _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type))
  ) : map_(exts), ctr_(map_.required_span_size())
  { }

  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdarray, (const mapping_type& m), ,
    /* requires */ (_MDSPAN_TRAIT(is_constructible, container_type, size_t))
  ) : map_(m), ctr_(map_.required_span_size())
  { }

  // Constructors for array
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      _MDSPAN_TRAIT(is_constructible, extents_type, SizeTypes...) &&
      _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type) &&
      _MDSPAN_TRAIT(is_same, container_type, array<ElementType,size_of_extents<Extents>::value()>)
    )
  )
  MDSPAN_INLINE_FUNCTION
  explicit constexpr mdarray(SizeTypes... dynamic_extents)
    : map_(extents_type(dynamic_extents...)), ctr_()
  { }

  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdarray, (const extents_type& exts), ,
    /* requires */ (_MDSPAN_TRAIT(is_constructible, mapping_type, extents_type) &&
                    _MDSPAN_TRAIT(is_same, container_type, array<ElementType,size_of_extents<Extents>::value()>))
  ) : map_(exts), ctr_()
  { }

  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdarray, (const mapping_type& m), ,
    /* requires */ (_MDSPAN_TRAIT(is_same, container_type, array<ElementType,size_of_extents<Extents>::value()>))
  ) : map_(m), ctr_()
  { }

  // Constructors from container
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      _MDSPAN_TRAIT(is_constructible, extents_type, SizeTypes...) &&
      _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type)
    )
  )
  MDSPAN_INLINE_FUNCTION
  explicit constexpr mdarray(const container_type& ctr, SizeTypes... dynamic_extents)
    : map_(extents_type(dynamic_extents...)), ctr_(ctr)
  { assert(ctr.size() >= map_.required_span_size()); }


  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdarray, (const container_type& ctr, const extents_type& exts), ,
    /* requires */ (_MDSPAN_TRAIT(is_constructible, mapping_type, extents_type))
  ) : map_(exts), ctr_(ctr)
  { assert(ctr.size() >= map_.required_span_size()); }

  constexpr mdarray(const container_type& ctr, const mapping_type& m)
    : map_(m), ctr_(ctr)
  { assert(ctr.size() >= map_.required_span_size()); }


  // Constructors from container
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      _MDSPAN_TRAIT(is_constructible, extents_type, SizeTypes...) &&
      _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type)
    )
  )
  MDSPAN_INLINE_FUNCTION
  explicit constexpr mdarray(container_type&& ctr, SizeTypes... dynamic_extents)
    : map_(extents_type(dynamic_extents...)), ctr_(std::move(ctr))
  { assert(ctr_.size() >= map_.required_span_size()); }


  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdarray, (container_type&& ctr, const extents_type& exts), ,
    /* requires */ (_MDSPAN_TRAIT(is_constructible, mapping_type, extents_type))
  ) : map_(exts), ctr_(std::move(ctr))
  { assert(ctr_.size() >= map_.required_span_size()); }

  constexpr mdarray(container_type&& ctr, const mapping_type& m)
    : map_(m), ctr_(std::move(ctr))
  { assert(ctr_.size() >= map_.required_span_size()); }



  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType, class OtherExtents, class OtherLayoutPolicy, class OtherContainer,
    /* requires */ (
      _MDSPAN_TRAIT(is_constructible, mapping_type, typename OtherLayoutPolicy::template mapping<OtherExtents>) &&
      _MDSPAN_TRAIT(is_constructible, container_type, OtherContainer)
    )
  )
  MDSPAN_INLINE_FUNCTION
  constexpr mdarray(const mdarray<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainer>& other)
    : map_(other.mapping()), ctr_(other.container())
  {
    static_assert(is_constructible_v<extents_type, OtherExtents>);
  }

  MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~mdarray() = default;

  //--------------------------------------------------------------------------------
  // [mdspan.basic.mapping], mdspan mapping domain multidimensional index to access codomain element

  #if MDSPAN_USE_BRACKET_OPERATOR
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      extents_type::rank() == sizeof...(SizeTypes)
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr const_reference operator[](SizeTypes... indices) const noexcept
  {
    return ctr_[map_(size_type(indices)...)];
  }

  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      extents_type::rank() == sizeof...(SizeTypes)
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator[](SizeTypes... indices) noexcept
  {
    return ctr_[map_(size_type(indices)...)];
  }
  #endif

#if 0
  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, SizeType, size_type) &&
      N == extents_type::rank()
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr const_reference operator[](const array<SizeType, N>& indices) const noexcept
  {
    return __impl::template __callop<reference>(*this, indices);
  }

  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, SizeType, size_type) &&
      N == extents_type::rank()
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator[](const array<SizeType, N>& indices) noexcept
  {
    return __impl::template __callop<reference>(*this, indices);
  }
#endif


  #if MDSPAN_USE_PAREN_OPERATOR
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      extents_type::rank() == sizeof...(SizeTypes)
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr const_reference operator()(SizeTypes... indices) const noexcept
  {
    return ctr_[map_(size_type(indices)...)];
  }
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      extents_type::rank() == sizeof...(SizeTypes)
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator()(SizeTypes... indices) noexcept
  {
    return ctr_[map_(size_type(indices)...)];
  }

#if 0
  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, SizeType, size_type) &&
      N == extents_type::rank()
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr const_reference operator()(const array<SizeType, N>& indices) const noexcept
  {
    return __impl::template __callop<reference>(*this, indices);
  }

  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, SizeType, size_type) &&
      N == extents_type::rank()
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator()(const array<SizeType, N>& indices) noexcept
  {
    return __impl::template __callop<reference>(*this, indices);
  }
#endif
  #endif

  MDSPAN_INLINE_FUNCTION constexpr pointer data() noexcept { return ctr_.data(); };
  MDSPAN_INLINE_FUNCTION constexpr const_pointer data() const noexcept { return ctr_.data(); };
  MDSPAN_INLINE_FUNCTION constexpr container_type& container() noexcept { return ctr_; };
  MDSPAN_INLINE_FUNCTION constexpr const container_type& container() const noexcept { return ctr_; };

  //--------------------------------------------------------------------------------
  // [mdspan.basic.domobs], mdspan observers of the domain multidimensional index space

  MDSPAN_INLINE_FUNCTION static constexpr size_t rank() noexcept { return extents_type::rank(); }
  MDSPAN_INLINE_FUNCTION static constexpr size_t rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  MDSPAN_INLINE_FUNCTION static constexpr size_type static_extent(size_t r) noexcept { return extents_type::static_extent(r); }

  MDSPAN_INLINE_FUNCTION constexpr extents_type extents() const noexcept { return map_.extents(); };
  MDSPAN_INLINE_FUNCTION constexpr size_type extent(size_t r) const noexcept { return map_.extents().extent(r); };
  MDSPAN_INLINE_FUNCTION constexpr size_type size() const noexcept {
//    return __impl::__size(*this);
    return ctr_.size();
  };


  //--------------------------------------------------------------------------------
  // [mdspan.basic.obs], mdspan observers of the mapping

  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return mapping_type::is_always_unique(); };
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_contiguous() noexcept { return mapping_type::is_always_contiguous(); };
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return mapping_type::is_always_strided(); };

  MDSPAN_INLINE_FUNCTION constexpr mapping_type mapping() const noexcept { return map_; };
  MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept { return map_.is_unique(); };
  MDSPAN_INLINE_FUNCTION constexpr bool is_contiguous() const noexcept { return map_.is_contiguous(); };
  MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept { return map_.is_strided(); };
  MDSPAN_INLINE_FUNCTION constexpr size_type stride(size_t r) const { return map_.stride(r); };

private:
  mapping_type map_;
  container_type ctr_;

  template <class, class, class, class>
  friend class mdarray;
};


} // end namespace experimental
} // end namespace std

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

#include <experimental/mdarray>
#include <vector>

#include <gtest/gtest.h>
#include "offload_utils.hpp"

namespace stdex = std::experimental;
_MDSPAN_INLINE_VARIABLE constexpr auto dyn = stdex::dynamic_extent;


void test_mdarray_ctor_data_carray() {
  size_t* errors = allocate_array<size_t>(1);
  errors[0] = 0;

  dispatch([=] _MDSPAN_HOST_DEVICE () {
    stdex::mdarray<int, stdex::extents<1>> m(stdex::extents<1>{});
    __MDSPAN_DEVICE_ASSERT_EQ(m.rank(), 1);
    __MDSPAN_DEVICE_ASSERT_EQ(m.rank_dynamic(), 0);
    __MDSPAN_DEVICE_ASSERT_EQ(m.extent(0), 1);
    __MDSPAN_DEVICE_ASSERT_EQ(m.static_extent(0), 1);
    __MDSPAN_DEVICE_ASSERT_EQ(m.stride(0), 1);
    m.data()[0] = {42};
    auto val = __MDSPAN_OP(m,0);
    __MDSPAN_DEVICE_ASSERT_EQ(val, 42);
    __MDSPAN_DEVICE_ASSERT_EQ(m.is_contiguous(), true);
  });
  ASSERT_EQ(errors[0], 0);
  free_array(errors);
}

TEST(TestMdarrayCtorDataCArray, test_mdarray_ctor_data_carray) {
  __MDSPAN_TESTS_RUN_TEST(test_mdarray_ctor_data_carray())
}

// Construct from extents only
TEST(TestMdarrayCtorFromExtents, 0d_static) {
  stdex::mdarray<int, stdex::extents<>> m(stdex::extents<>{});
  ASSERT_EQ(m.rank(), 0);
  ASSERT_EQ(m.rank_dynamic(), 0);
  m.data()[0] = 42;
  ASSERT_EQ(__MDSPAN_OP(m), 42);
  ASSERT_TRUE(m.is_contiguous());
}

// Construct from sizes only
TEST(TestMdarrayCtorFromSizes, 1d_static) {
  stdex::mdarray<int, stdex::extents<1>> m(1);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  m.data()[0] = 42;
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromSizes, 2d_static) {
  stdex::mdarray<int, stdex::extents<2,3>> m(2,3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  m.data()[0] = 42;
  m.data()[5] = 41;
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromSizes, 1d_dynamic) {
  stdex::mdarray<int, stdex::dextents<1>> m(1);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 1);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  m.data()[0] = 42;
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromSizes, 2d_dynamic) {
  stdex::mdarray<int, stdex::dextents<2>> m(2,3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  m.data()[0] = 42;
  m.data()[5] = 41;
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromSizes, 2d_mixed) {
  stdex::mdarray<int, stdex::extents<2,stdex::dynamic_extent>> m(3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 1);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  m.data()[0] = 42;
  m.data()[5] = 41;
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_TRUE(m.is_contiguous());
}

// Construct from container + sizes
TEST(TestMdarrayCtorFromContainerSizes, 1d_static) {
  std::array<int, 1> d{42};
  stdex::mdarray<int, stdex::extents<1>> m(d,1);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_NE(m.data(),d.data());
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromContainerSizes, 2d_static) {
  std::array<int, 6> d{42,1,2,3,4,41};
  stdex::mdarray<int, stdex::extents<2,3>> m(d,2,3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_NE(m.data(),d.data());
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromContainerSizes, 1d_dynamic) {
  std::vector<int> d{42};
  stdex::mdarray<int, stdex::dextents<1>> m(d,1);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 1);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_NE(m.data(),d.data());
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromContainerSizes, 2d_dynamic) {
  std::vector<int> d{42,1,2,3,4,41};
  stdex::mdarray<int, stdex::dextents<2>> m(d,2,3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_NE(m.data(),d.data());
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromContainerSizes, 2d_mixed) {
  std::vector<int> d{42,1,2,3,4,41};
  stdex::mdarray<int, stdex::extents<2,stdex::dynamic_extent>> m(d,3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 1);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_NE(m.data(),d.data());
  ASSERT_TRUE(m.is_contiguous());
}

// Construct from move container + sizes
TEST(TestMdarrayCtorFromMoveContainerSizes, 1d_static) {
  std::array<int, 1> d{42};
  stdex::mdarray<int, stdex::extents<1>> m(std::move(d),1);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 2d_static) {
  std::array<int, 6> d{42,1,2,3,4,41};
  stdex::mdarray<int, stdex::extents<2,3>> m(std::move(d),2,3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 1d_dynamic) {
  std::vector<int> d{42};
  auto ptr = d.data();
  stdex::mdarray<int, stdex::dextents<1>> m(std::move(d),1);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 1);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_EQ(m.data(),ptr);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 2d_dynamic) {
  std::vector<int> d{42,1,2,3,4,41};
  auto ptr = d.data();
  stdex::mdarray<int, stdex::dextents<2>> m(std::move(d),2,3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_EQ(m.data(),ptr);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 2d_mixed) {
  std::vector<int> d{42,1,2,3,4,41};
  auto ptr = d.data();
  stdex::mdarray<int, stdex::extents<2,stdex::dynamic_extent>> m(std::move(d),3);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 1);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 3);
  ASSERT_EQ(m.stride(0), 3);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0, 0), 42);
  ASSERT_EQ(__MDSPAN_OP(m, 1, 2), 41);
  ASSERT_EQ(m.data(),ptr);
  ASSERT_TRUE(m.is_contiguous());
}

// Construct from container only
TEST(TestMdarrayCtorDataStdArray, test_mdarray_ctor_data_carray) {
  std::array<int, 1> d = {42};
  stdex::mdarray<int, stdex::extents<1>> m(d);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorDataVector, test_mdarray_ctor_data_carray) {
  std::vector<int> d = {42};
  stdex::mdarray<int, stdex::extents<1>, stdex::layout_right, std::vector<int>> m(d);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCtorExtentsStdArrayConvertibleToSizeT, test_mdarray_ctor_extents_std_array_convertible_to_size_t) {
  std::vector<int> d{42, 17, 71, 24};
  std::array<int, 2> e{2, 2};
  stdex::mdarray<int, stdex::dextents<2>> m(d, e);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 2);
  ASSERT_EQ(m.stride(0), 2);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_contiguous());
}


TEST(TestMdarrayListInitializationLayoutLeft, test_mdarray_list_initialization_layout_left) {
  std::vector<int> d(16*32);
  auto ptr = d.data();
  stdex::mdarray<int, stdex::extents<dyn, dyn>, stdex::layout_left> m{std::move(d), 16, 32};
  ASSERT_EQ(m.data(), ptr);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 16);
  ASSERT_TRUE(m.is_contiguous());
}


TEST(TestMdarrayListInitializationLayoutRight, test_mdarray_list_initialization_layout_right) {
  std::vector<int> d(16*32);
  auto ptr = d.data();
  stdex::mdarray<int, stdex::extents<dyn, dyn>, stdex::layout_right> m{std::move(d), 16, 32};
  ASSERT_EQ(m.data(), ptr);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 32);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayListInitializationLayoutStride, test_mdarray_list_initialization_layout_stride) {
  std::vector<int> d(32*128);
  auto ptr = d.data();
  stdex::mdarray<int, stdex::extents<dyn, dyn>, stdex::layout_stride> m{std::move(d), {stdex::dextents<2>{16, 32}, std::array<std::size_t, 2>{1, 128}}};
  ASSERT_EQ(m.data(), ptr);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 128);
  ASSERT_FALSE(m.is_contiguous());
}

#if 0

#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
TEST(TestMdarrayCTAD, extents_pack) {
  std::array<int, 1> d{42};
  stdex::mdarray m(d.data(), 64, 128);
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCTAD, ctad_pointer) {
  std::array<int,5> d = {1,2,3,4,5};
  stdex::mdarray m(d.data());
  static_assert(std::is_same<decltype(m)::element_type,int>::value);
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 0);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCTAD, ctad_carray) {
  int data[5] = {1,2,3,4,5};
  stdex::mdarray m(data);
  static_assert(std::is_same<decltype(m)::element_type,int>::value);
  ASSERT_EQ(m.data(), &data[0]);
  #ifdef  _MDSPAN_USE_P2554
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.static_extent(0), 5);
  ASSERT_EQ(m.extent(0), 5);
  ASSERT_EQ(__MDSPAN_OP(m, 2), 3);
  #else
  ASSERT_EQ(m.rank(), 0);
  ASSERT_EQ(m.rank_dynamic(), 0);
  #endif
  ASSERT_TRUE(m.is_contiguous());


  stdex::mdarray m2(data, 3);
  static_assert(std::is_same<decltype(m2)::element_type,int>::value);
  ASSERT_EQ(m2.data(), &data[0]);
  ASSERT_EQ(m2.rank(), 1);
  ASSERT_EQ(m2.rank_dynamic(), 1);
  ASSERT_EQ(m2.extent(0), 3);
  ASSERT_TRUE(m2.is_contiguous());
  ASSERT_EQ(__MDSPAN_OP(m2, 2), 3);
}

TEST(TestMdarrayCTAD, ctad_const_carray) {
  const int data[5] = {1,2,3,4,5};
  stdex::mdarray m(data);
  static_assert(std::is_same<decltype(m)::element_type,const int>::value);
  ASSERT_EQ(m.data(), &data[0]);
  #ifdef  _MDSPAN_USE_P2554
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.static_extent(0), 5);
  ASSERT_EQ(m.extent(0), 5);
  ASSERT_EQ(__MDSPAN_OP(m, 2), 3);
  #else
  ASSERT_EQ(m.rank(), 0);
  ASSERT_EQ(m.rank_dynamic(), 0);
  #endif
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCTAD, extents_object) {
  std::array<int, 1> d{42};
  stdex::mdarray m{d.data(), stdex::extents{64, 128}};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCTAD, extents_std_array) {
  std::array<int, 1> d{42};
  stdex::mdarray m{d.data(), std::array{64, 128}};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdarrayCTAD, layout_left) {
  std::array<int, 1> d{42};

  stdex::mdarray m0{d.data(), stdex::layout_left::mapping{stdex::extents{16, 32}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 1);
  ASSERT_EQ(m0.stride(1), 16);
  ASSERT_TRUE(m0.is_contiguous());

// TODO: Perhaps one day I'll get this to work.
/*
  stdex::mdarray m1{d.data(), stdex::layout_left::mapping{{16, 32}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 16);
  ASSERT_TRUE(m1.is_contiguous());
*/
}

TEST(TestMdarrayCTAD, layout_right) {
  std::array<int, 1> d{42};

  stdex::mdarray m0{d.data(), stdex::layout_right::mapping{stdex::extents{16, 32}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 32);
  ASSERT_EQ(m0.stride(1), 1);
  ASSERT_TRUE(m0.is_contiguous());

// TODO: Perhaps one day I'll get this to work.
/*
  stdex::mdarray m1{d.data(), stdex::layout_right::mapping{{16, 32}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 32);
  ASSERT_EQ(m1.stride(1), 1);
  ASSERT_TRUE(m1.is_contiguous());
*/
}

TEST(TestMdarrayCTAD, layout_stride) {
  std::array<int, 1> d{42};

  stdex::mdarray m0{d.data(), stdex::layout_stride::mapping{stdex::extents{16, 32}, std::array{1, 128}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 1);
  ASSERT_EQ(m0.stride(1), 128);
  ASSERT_FALSE(m0.is_contiguous());

  /* 
  stdex::mdarray m1{d.data(), stdex::layout_stride::mapping{stdex::extents{16, 32}, stdex::extents{1, 128}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 128);
  ASSERT_FALSE(m1.is_contiguous());
  */

// TODO: Perhaps one day I'll get this to work.
/*
  stdex::mdarray m2{d.data(), stdex::layout_stride::mapping{{16, 32}, {1, 128}}};
  ASSERT_EQ(m2.data(), d.data());
  ASSERT_EQ(m2.rank(), 2);
  ASSERT_EQ(m2.rank_dynamic(), 2);
  ASSERT_EQ(m2.extent(0), 16);
  ASSERT_EQ(m2.extent(1), 32);
  ASSERT_EQ(m2.stride(0), 1);
  ASSERT_EQ(m2.stride(1), 128);
  ASSERT_FALSE(m2.is_contiguous());
*/
}
#endif

#endif

// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <cstdint>
namespace ck_tile {
struct Int8CachedWeightPackProblem {
 static constexpr index_t NTile=16,KTile=64,VectorBytes=16,Threads=256;
};
struct Int8CachedWeightPackArgs {
 const int8_t* source;
 int8_t* packed;
 int32_t n,k,stride;
};
template<class Problem=Int8CachedWeightPackProblem> struct Int8CachedWeightPackKernel {
 CK_TILE_DEVICE void operator()(const Int8CachedWeightPackArgs* args) const {
  const auto a=args[blockIdx.z];const uint32_t lane=get_thread_id()%64,wave=get_thread_id()/64;
  const uint64_t nt=static_cast<uint64_t>(blockIdx.x)*(Problem::Threads/64)+wave,kt=blockIdx.y;
  if(nt>=static_cast<uint32_t>(a.n/16))return;
  const uint64_t ni=lane&15,kg=lane>>4;
  const uint64_t v=(kt*(a.n/16)+nt)*64+lane;
  using Vec=int32_t __attribute__((ext_vector_type(4)));
  const auto* source=a.source+(nt*16+ni)*a.stride+kt*64+kg*16;
  *reinterpret_cast<Vec*>(a.packed+v*16)=*reinterpret_cast<const Vec*>(source);
 }
};
}

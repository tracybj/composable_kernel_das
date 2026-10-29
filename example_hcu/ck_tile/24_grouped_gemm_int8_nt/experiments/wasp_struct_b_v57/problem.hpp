// SPDX-License-Identifier: MIT
#pragma once
#include "../hoisted_copy_v55/problem.hpp"
namespace ck_tile {
template<bool Transpose_=true,bool CacheSwizzle_=true,int Group_=4,bool Slab_=false>
struct GroupedInt8NtStructBProblem : GroupedInt8NtHoistedCopyProblem<true,Group_,false,false>
{
    static constexpr bool Transpose=Transpose_,CacheSwizzle=CacheSwizzle_,SlabCopy=Slab_;
};
}

// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
#include "m128/problem.hpp"
#include "m64/problem.hpp"
#include "m32/problem.hpp"
#include "wide_n/problem.hpp"
namespace ck_tile::int8_nt_instances {
// Stable route handles identify precompiled geometry/policy, never workload IDs.
using RawM256N256 = GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,false,1>;
using RawM256N256Record1024 = GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,1024,false,false,1>;
using RawTransposeM256N256 = GroupedInt8NtWaspCachedProblem<true,8,4,4,true,true,0,false,false,1>;
using RawTransposeM256N256Record1024 = GroupedInt8NtWaspCachedProblem<true,8,4,4,true,true,1024,false,false,1>;
using PackedM256N256 = GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,true,0>;
using PackedM128N256 = GroupedInt8NtWaspSplitMProblem<false,8,4,2,true,true,0,false,true,0,8>;
using PackedM128N128 = GroupedInt8NtWaspSplitMProblem<false,4,1,4,true,true,0,false,true,0,4>;
using PackedM64N256 = GroupedInt8NtWaspSmallMProblem<false,8,1,2,true,true,0,false,true,1,8>;
using PackedM64N64 = GroupedInt8NtWaspSmallMProblem<false,2,1,2,true,true,0,false,true,1,2>;
using PackedM32N256 = GroupedInt8NtWaspM32Problem<false,8,1,2,true,true,0,false,true,1,8>;
using PackedM32N64 = GroupedInt8NtWaspM32Problem<false,2,1,1,true,true,0,false,true,1,2>;
using PackedM256N384 = GroupedInt8NtWaspPublicWideNProblem<false,12,4,2,true,true,0,false,true,0,12>;
}

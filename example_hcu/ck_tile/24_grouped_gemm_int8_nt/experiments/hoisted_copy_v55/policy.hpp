// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
// Same raw-buffer -> LDS ABI as hcu_async_buffer_load_asm_impl<int8_t,16>.
// One wave-uniform LDS base and eight precomputed vector offsets replace
// eight separate m0 derivations. Each issue advances the LDS base by4096.
struct GroupedInt8NtHoistedCopyPolicy
{
    struct Offsets {index_t v[8];};
    CK_TILE_DEVICE static void Copy(uint32_t wave_m0,const int32x4_t& ar,const Offsets& o)
    {
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "buffer_load_dwordx4 %1, %9, 0 offen lds\n\t"
            "s_add_u32 m0, m0, 4096\n\t"
            "buffer_load_dwordx4 %2, %9, 0 offen lds\n\t"
            "s_add_u32 m0, m0, 4096\n\t"
            "buffer_load_dwordx4 %3, %9, 0 offen lds\n\t"
            "s_add_u32 m0, m0, 4096\n\t"
            "buffer_load_dwordx4 %4, %9, 0 offen lds\n\t"
            "s_add_u32 m0, m0, 4096\n\t"
            "buffer_load_dwordx4 %5, %9, 0 offen lds\n\t"
            "s_add_u32 m0, m0, 4096\n\t"
            "buffer_load_dwordx4 %6, %9, 0 offen lds\n\t"
            "s_add_u32 m0, m0, 4096\n\t"
            "buffer_load_dwordx4 %7, %9, 0 offen lds\n\t"
            "s_add_u32 m0, m0, 4096\n\t"
            "buffer_load_dwordx4 %8, %9, 0 offen lds\n\t"
            : : "s"(wave_m0),"v"(o.v[0]),"v"(o.v[1]),"v"(o.v[2]),"v"(o.v[3]),
                "v"(o.v[4]),"v"(o.v[5]),"v"(o.v[6]),"v"(o.v[7]),"s"(ar) : "memory");
    }
    CK_TILE_DEVICE static void Advance(int32x4_t& ar)
    {
        const uint32_t old=static_cast<uint32_t>(ar[0]),next=old+128u;
        const uint32_t hi=static_cast<uint32_t>(ar[1])+static_cast<uint32_t>(next<old);
        ar[0]=bit_cast<int32_t>(next);ar[1]=bit_cast<int32_t>(hi);
    }
};
}

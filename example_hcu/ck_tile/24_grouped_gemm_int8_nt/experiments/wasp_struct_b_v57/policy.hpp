// SPDX-License-Identifier: MIT
// Structured-buffer/cache-swizzle and four-read form adapted from Hygon's
// gemm_pipeline_ag_bg_cr_wasp.hpp (MIT, copyright2026 Hygon Info Technologies).
#pragma once
#include "problem.hpp"
#include "../bypass_b_ordered_v52/pipeline.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtStructBPolicy
{
    using BTile=typename GroupedInt8NtBypassBOrderedPipeline<P>::BTile;
    struct Offsets {index_t v[8];};
    CK_TILE_DEVICE static int32x4_t BResource(const void* ptr,uint32_t pitch,uint32_t bytes)
    {
        const uint64_t address=reinterpret_cast<uint64_t>(ptr);
        uint32_t hi=static_cast<uint32_t>(address>>32)|((pitch&0x3fffu)<<16);
#if defined(__gfx936__) || defined(__gfx938__)
        if constexpr(P::CacheSwizzle)
            if(pitch%64==0 && pitch>=64 && (((pitch/64)&(pitch/64-1))==0))hi|=1u<<30;
#endif
        return {bit_cast<int32_t>(static_cast<uint32_t>(address)),bit_cast<int32_t>(hi),
                bit_cast<int32_t>(bytes),0x00020000};
    }
    CK_TILE_DEVICE static BTile LoadB(const int32x4_t& br,uint64_t p0,uint64_t p1,index_t kb)
    {
        BTile tile;
        asm volatile(
            "buffer_load_dwordx4 %0, %4, %6, %7 idxen offen offset:0\n\t"
            "buffer_load_dwordx4 %1, %4, %6, %7 idxen offen offset:64\n\t"
            "buffer_load_dwordx4 %2, %5, %6, %7 idxen offen offset:0\n\t"
            "buffer_load_dwordx4 %3, %5, %6, %7 idxen offen offset:64\n"
            : "=&v"(tile.values[0][0]),"=&v"(tile.values[1][0]),
              "=&v"(tile.values[0][1]),"=&v"(tile.values[1][1])
            : "v"(p0),"v"(p1),"s"(br),"s"(kb) : "memory");
        return tile;
    }
    CK_TILE_DEVICE static void CopyA(uint32_t wave_m0,const int32x4_t& ar,const Offsets& o,index_t kb)
    {
        constexpr index_t step=P::SlabCopy?1024:4096;
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "buffer_load_dwordx4 %1, %9, %10 offen lds\n\t"
            "s_add_u32 m0, m0, %11\n\t"
            "buffer_load_dwordx4 %2, %9, %10 offen lds\n\t"
            "s_add_u32 m0, m0, %11\n\t"
            "buffer_load_dwordx4 %3, %9, %10 offen lds\n\t"
            "s_add_u32 m0, m0, %11\n\t"
            "buffer_load_dwordx4 %4, %9, %10 offen lds\n\t"
            "s_add_u32 m0, m0, %11\n\t"
            "buffer_load_dwordx4 %5, %9, %10 offen lds\n\t"
            "s_add_u32 m0, m0, %11\n\t"
            "buffer_load_dwordx4 %6, %9, %10 offen lds\n\t"
            "s_add_u32 m0, m0, %11\n\t"
            "buffer_load_dwordx4 %7, %9, %10 offen lds\n\t"
            "s_add_u32 m0, m0, %11\n\t"
            "buffer_load_dwordx4 %8, %9, %10 offen lds\n"
            : : "s"(wave_m0),"v"(o.v[0]),"v"(o.v[1]),"v"(o.v[2]),"v"(o.v[3]),
                "v"(o.v[4]),"v"(o.v[5]),"v"(o.v[6]),"v"(o.v[7]),"s"(ar),"s"(kb),"n"(step) : "memory");
    }
};
}

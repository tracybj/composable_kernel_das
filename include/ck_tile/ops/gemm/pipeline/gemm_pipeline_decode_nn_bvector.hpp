// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_wave.hpp"

// Isolated B-layout experiment. A, output tiles, K32, Depth2 and epilogue are
// shared with the scalar control. Width2 retains the scalar lane order and
// reads aligned N-pairs (neighbor lanes request the same word), selecting the
// required BF16 directly with v_perm. Width4 cooperatively transposes registers.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
namespace ck_tile {
namespace decode_nn_wave {
using U32x2 = uint32_t __attribute__((ext_vector_type(2)));
template<int Width>
struct VectorBPolicy
{
    static_assert(Width == 2 || Width == 4);
    using Registers = std::conditional_t<Width == 2, U32x4, U32x2>;
    static constexpr int Loads = Width == 2 ? 16 : 4;
    CK_TILE_DEVICE static uint32_t offset(const Params& p, uint32_t lane, int wave)
    {
        const uint32_t row = (lane >> 4) * 4u + (Width == 2 ? 0u : (lane & 3u));
        const uint32_t col = (uint32_t)p.n0 + (uint32_t)wave * 32u +
                             ((lane & 15u) & ~(Width - 1u));
        return (row * (uint32_t)p.ldb + col) * 2u;
    }
    template<int Pending>
    CK_TILE_DEVICE static void wait(Registers (&b)[2][2])
    {
        // Values remain LLVM-tracked VMEM results: do not substitute opaque
        // asynchronous asm loads whose results could be copied before ready.
        asm volatile("s_waitcnt vmcnt(%4)\n"
                     : "+v"(b[0][0]), "+v"(b[0][1]), "+v"(b[1][0]), "+v"(b[1][1])
                     : "n"(Pending) : "memory");
    }
    template<int Rows, int Slot>
    CK_TILE_DEVICE static void load(Srd a_srd, const uint64_t (&a_pair)[Rows], Srd b_srd,
                                    uint32_t b_offset, uint32_t wave_base, int ldb,
                                    int k, Registers (&b)[2][2])
    {
        static_for<0, Rows, 1>{}([&](auto index) {
            constexpr int m = decltype(index)::value;
            load_a(a_srd, a_pair[m], (uint32_t)k * 2u,
                   wave_base + (Slot * Rows + m) * 1024u);
        });
        static_for<0, 4, 1>{}([&](auto index) {
            constexpr int sub = decltype(index)::value / 2;
            constexpr int n = decltype(index)::value % 2;
            const uint32_t base = (uint32_t)(k + sub * 16) * (uint32_t)ldb * 2u + n * 32u;
            if constexpr(Width == 2)
            {
                const auto q0 = llvm_amdgcn_raw_buffer_load_i32(
                    bit_cast<int32x4_t>(b_srd), (index_t)b_offset, (index_t)base, 0);
                const auto q1 = llvm_amdgcn_raw_buffer_load_i32(
                    bit_cast<int32x4_t>(b_srd), (index_t)b_offset,
                    (index_t)(base + (uint32_t)ldb * 2u), 0);
                const auto q2 = llvm_amdgcn_raw_buffer_load_i32(
                    bit_cast<int32x4_t>(b_srd), (index_t)b_offset,
                    (index_t)(base + (uint32_t)ldb * 4u), 0);
                const auto q3 = llvm_amdgcn_raw_buffer_load_i32(
                    bit_cast<int32x4_t>(b_srd), (index_t)b_offset,
                    (index_t)(base + (uint32_t)ldb * 6u), 0);
                b[sub][n] = Registers{(uint32_t)q0, (uint32_t)q1, (uint32_t)q2, (uint32_t)q3};
            }
            else
            {
                b[sub][n] = bit_cast<Registers>(llvm_amdgcn_raw_buffer_load_i32x2(
                    bit_cast<int32x4_t>(b_srd), (index_t)b_offset, (index_t)base, 0));
            }
        });
    }
    CK_TILE_DEVICE static uint32_t exchange_pair(uint32_t value)
    {
        // quad_perm:[1,0,3,2] exchanges lane XOR 1 without LDS storage.
        return (uint32_t)__builtin_amdgcn_mov_dpp((int)value, 0xb1, 0xf, 0xf, false);
    }
    CK_TILE_DEVICE static uint32_t pack_pair(uint32_t value, uint32_t lane)
    {
        const uint32_t other = exchange_pair(value);
        // v_perm indexes src1 bytes 0..3 and src0 bytes 4..7.
        const uint32_t mask = (lane & 1u) ? 0x03020706u : 0x05040100u;
        return (uint32_t)__builtin_amdgcn_perm((int)other, (int)value, (int)mask);
    }
    CK_TILE_DEVICE static Bf16x4 rearrange(Registers raw, uint32_t lane)
    {
        if constexpr(Width == 2)
        {
            // A dword holds two adjacent N values for one K. Select the low
            // or high BF16 from each row and pack K pairs, with no DPP and no
            // uint16 narrowing before the prefetch values cross loop edges.
            const uint32_t mask = (lane & 1u) ? 0x07060302u : 0x05040100u;
            const U32x2 packed{
                (uint32_t)__builtin_amdgcn_perm((int)raw[1], (int)raw[0], (int)mask),
                (uint32_t)__builtin_amdgcn_perm((int)raw[3], (int)raw[2], (int)mask)};
            return bit_cast<Bf16x4>(packed);
        }
        else
        {
            // First transpose the two N-pairs across lane XOR 2, then transpose
            // adjacent K rows with pack_pair. All lanes execute each exchange.
            const uint32_t other0 = (uint32_t)__builtin_amdgcn_mov_dpp(
                (int)raw[0], 0x4e, 0xf, 0xf, false); // quad_perm:[2,3,0,1]
            const uint32_t other1 = (uint32_t)__builtin_amdgcn_mov_dpp(
                (int)raw[1], 0x4e, 0xf, 0xf, false);
            const uint32_t first = (lane & 2u) ? other1 : raw[0];
            const uint32_t second = (lane & 2u) ? raw[1] : other0;
            raw = Registers{first, second};
            return bit_cast<Bf16x4>(Registers{pack_pair(raw[0], lane), pack_pair(raw[1], lane)});
        }
    }
    template<int Rows>
    CK_TILE_DEVICE static void compute(const Registers (&b)[2][2], uint32_t slot,
                                       uint32_t lane, F32x4 (&c)[Rows][2])
    {
        static_for<0, 2, 1>{}([&](auto step) {
            constexpr int sub = decltype(step)::value;
            Bf16x4 bv[2];
            static_for<0, 2, 1>{}([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                bv[n] = rearrange(b[sub][n], lane);
            });
            static_for<0, Rows, 1>{}([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const uint32_t row = lane & 15u;
                const uint32_t byte = ((lane >> 4) * 4u + sub * 16u) * 2u;
                const uint32_t address = slot + m * 1024u + row * 64u +
                    (((byte >> 4) ^ (row & 3u)) << 4) + (byte & 15u);
                const auto av = read_a(address);
                c[m][0] = __builtin_hcu_mmac_f32_16x16x16_bf16(av, bv[0], c[m][0]);
                c[m][1] = __builtin_hcu_mmac_f32_16x16x16_bf16(av, bv[1], c[m][1]);
            });
        });
    }
};
} // namespace decode_nn_wave
} // namespace ck_tile
#pragma clang diagnostic pop

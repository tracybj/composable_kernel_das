// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_policy.hpp"
#include "../include/grouped_int8_nt_scale_args.hpp"
namespace ck_tile {
namespace int8_nt_bf16_bits {
constexpr uint32_t DiscardedBits          = 16;
constexpr uint32_t RoundToNearestEvenBias = 0x7fffU;
constexpr uint32_t Fp32MagnitudeMask      = 0x7fffffffU;
constexpr uint32_t Fp32Infinity           = 0x7f800000U;
constexpr uint32_t Fp32HighHalfMask       = 0xffff0000U;
constexpr uint32_t Fp32QuietNanBit        = 0x00400000U;
constexpr uint32_t Bf16QuietNanBit        = 0x40U;
} // namespace int8_nt_bf16_bits

// Keep the FP32 rounding boundary before the selected 16-bit conversion.
// Explicit select avoids a divergent NaN-handling region for every BF16 value.
template <bool BFloat>
struct Int8NtDequantConvert
{
    CK_TILE_DEVICE static uint16_t run(int32_t value, float combined_scale)
    {
        float result = static_cast<float>(value) * combined_scale;
        asm volatile("" : "+v"(result));
        if constexpr(BFloat)
        {
            const uint32_t bits    = bit_cast<uint32_t>(result);
            uint32_t lsb, rounded;
            asm volatile("v_bfe_u32 %[lsb], %[bits], 16, 1\n\t"
                         "v_add3_u32 %[rounded], %[bits], %[bias], %[lsb]"
                         : [lsb] "=&v"(lsb), [rounded] "=&v"(rounded)
                         : [bits] "v"(bits), [bias] "s"(int8_nt_bf16_bits::RoundToNearestEvenBias));
            return static_cast<uint16_t>(rounded >> 16);
        }
        else
            return bit_cast<uint16_t>(type_convert<half_t>(result));
    }
};
// Scales use ORIGINAL output coordinates, including the logical-transpose route.
// mode strides are 0 for a scalar and 1 for an outer-vector scale.
template <bool BFloat>
struct Int8NtDequantStore
{
    Int8NtScaleArgs scale;
    CK_TILE_DEVICE uint16_t convert(int32_t value, index_t row, index_t col) const
    {
        const float sa = scale.a ? scale.a[row * scale.a_stride] : 1.0f;
        const float sb = scale.b ? scale.b[col * scale.b_stride] : 1.0f;
        // Match the installed BLASLt SAB epilogue:round the product of FP32
        // scales before applying it to the converted integer accumulator.
        return Int8NtDequantConvert<BFloat>::run(value, sa * sb);
    }
    CK_TILE_DEVICE void
    one(const UniversalGemmKernelArgs<>& a, index_t row, index_t col, int32_t value) const
    {
        static_cast<uint16_t*>(a.e_ptr)[static_cast<long long>(row) * a.stride_E + col] =
            convert(value, row, col);
    }
    CK_TILE_DEVICE void four(const UniversalGemmKernelArgs<>& a,
                             index_t row,
                             index_t col,
                             int32_t r0,
                             int32_t r1,
                             int32_t r2,
                             int32_t r3) const
    {
        using V        = uint16_t __attribute__((ext_vector_type(4)));
        const float sa =
            scale.a ? (scale.a_stride ? scale.a[row * scale.a_stride] : scale.a[0]) : 1.0f;
        float b0 = 1.0f, b1 = 1.0f, b2 = 1.0f, b3 = 1.0f;
        if(scale.b)
        {
            if(scale.b_stride == 1)
            {
                const float* sb = scale.b + col;
                b0              = sb[0];
                b1              = sb[1];
                b2              = sb[2];
                b3              = sb[3];
            }
            else if(scale.b_stride == 0)
            {
                b0 = b1 = b2 = b3 = scale.b[0];
            }
            else
            {
                b0 = scale.b[col * scale.b_stride];
                b1 = scale.b[(col + 1) * scale.b_stride];
                b2 = scale.b[(col + 2) * scale.b_stride];
                b3 = scale.b[(col + 3) * scale.b_stride];
            }
        }
        const V v{Int8NtDequantConvert<BFloat>::run(r0, sa * b0),
                  Int8NtDequantConvert<BFloat>::run(r1, sa * b1),
                  Int8NtDequantConvert<BFloat>::run(r2, sa * b2),
                  Int8NtDequantConvert<BFloat>::run(r3, sa * b3)};
        *reinterpret_cast<V*>(static_cast<uint16_t*>(a.e_ptr) +
                              static_cast<long long>(row) * a.stride_E + col) = v;
    }
};
// Transpose the 4x4 (lane-K-group, accumulator-component) matrix in registers.
CK_TILE_DEVICE void
transpose_mmac_4x4(index_t lane, index_t h, int32_t& r0, int32_t& r1, int32_t& r2, int32_t& r3)
{
    const bool b0 = (h & 1) != 0, b1 = (h & 2) != 0;
    int32_t t0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r0 : r1),
            t1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r2 : r3);
    r0         = b0 ? t0 : r0;
    r1         = b0 ? r1 : t0;
    r2         = b0 ? t1 : r2;
    r3         = b0 ? r3 : t1;
    t0         = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r0 : r2);
    t1         = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r1 : r3);
    r0         = b1 ? t0 : r0;
    r2         = b1 ? r2 : t0;
    r1         = b1 ? t1 : r1;
    r3         = b1 ? r3 : t1;
}

namespace int8_nt_paired_fast {
using FloatPair = ext_vector_t<float, 2>;
using Rsrc4     = ext_vector_t<int32_t, 4>;
using Output4   = ext_vector_t<uint32_t, 4>;

constexpr uint32_t PackRoundedBf16Halves = 0x07060302U;
constexpr uint32_t GatherLowHalves       = 0x05040100U;
constexpr uint32_t GatherHighHalves      = 0x07060302U;

CK_TILE_DEVICE FloatPair pk_mul(FloatPair values, FloatPair factors)
{
    FloatPair result;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(values), "v"(factors));
    return result;
}

CK_TILE_DEVICE float canon_factor(float f)
{
    const uint32_t bits = bit_cast<uint32_t>(f);
    const uint32_t nan =
        (bits & int8_nt_bf16_bits::Fp32HighHalfMask) | int8_nt_bf16_bits::Fp32QuietNanBit;
    uint32_t out;
    asm volatile("v_cmp_u_f32 vcc, %1, %1\n\t"
                 "v_cndmask_b32 %0, %1, %2, vcc"
                 : "=v"(out)
                 : "v"(f), "v"(nan)
                 : "vcc");
    return bit_cast<float>(out);
}

CK_TILE_DEVICE uint32_t bf16_lsb(uint32_t bits)
{
    uint32_t lsb;
    asm("v_bfe_u32 %0, %1, 16, 1" : "=v"(lsb) : "v"(bits));
    return lsb;
}

CK_TILE_DEVICE uint32_t bf16_add_rne(uint32_t bits, uint32_t lsb)
{
    uint32_t rounded;
    asm("v_add3_u32 %0, %1, %2, %3"
        : "=v"(rounded)
        : "v"(bits), "s"(int8_nt_bf16_bits::RoundToNearestEvenBias), "v"(lsb));
    return rounded;
}

CK_TILE_DEVICE uint32_t bf16_pack_halves(uint32_t r0, uint32_t r1)
{
    uint32_t packed;
    asm("v_perm_b32 %0, %1, %2, %3"
        : "=v"(packed)
        : "v"(r1), "v"(r0), "s"(PackRoundedBf16Halves));
    return packed;
}

CK_TILE_DEVICE uint32_t fp16_pack_halves(float f0, float f1)
{
    const uint16_t h0 = bit_cast<uint16_t>(type_convert<half_t>(f0));
    const uint16_t h1 = bit_cast<uint16_t>(type_convert<half_t>(f1));
    uint32_t packed;
    asm("v_pack_b32_f16 %0, %1, %2" : "=v"(packed) : "v"(h0), "v"(h1));
    return packed;
}

template <bool BFloat, class Vec4>
CK_TILE_DEVICE void scale_quad(const Vec4& first,
                               const Vec4& second,
                               const FloatPair (&factors)[4],
                               uint32_t& p0,
                               uint32_t& p1,
                               uint32_t& p2,
                               uint32_t& p3)
{
    const FloatPair m0 = pk_mul(FloatPair{static_cast<float>(first[0]), static_cast<float>(first[1])}, factors[0]);
    const FloatPair m1 = pk_mul(FloatPair{static_cast<float>(first[2]), static_cast<float>(first[3])}, factors[1]);
    const FloatPair m2 = pk_mul(FloatPair{static_cast<float>(second[0]), static_cast<float>(second[1])}, factors[2]);
    const FloatPair m3 = pk_mul(FloatPair{static_cast<float>(second[2]), static_cast<float>(second[3])}, factors[3]);
    if constexpr(BFloat)
    {
        const uint32_t b00 = bit_cast<uint32_t>(m0[0]), b01 = bit_cast<uint32_t>(m0[1]);
        const uint32_t b10 = bit_cast<uint32_t>(m1[0]), b11 = bit_cast<uint32_t>(m1[1]);
        const uint32_t b20 = bit_cast<uint32_t>(m2[0]), b21 = bit_cast<uint32_t>(m2[1]);
        const uint32_t b30 = bit_cast<uint32_t>(m3[0]), b31 = bit_cast<uint32_t>(m3[1]);

        const uint32_t l00 = bf16_lsb(b00), l01 = bf16_lsb(b01);
        const uint32_t l10 = bf16_lsb(b10), l11 = bf16_lsb(b11);
        const uint32_t l20 = bf16_lsb(b20), l21 = bf16_lsb(b21);
        const uint32_t l30 = bf16_lsb(b30), l31 = bf16_lsb(b31);

        const uint32_t r00 = bf16_add_rne(b00, l00), r01 = bf16_add_rne(b01, l01);
        const uint32_t r10 = bf16_add_rne(b10, l10), r11 = bf16_add_rne(b11, l11);
        const uint32_t r20 = bf16_add_rne(b20, l20), r21 = bf16_add_rne(b21, l21);
        const uint32_t r30 = bf16_add_rne(b30, l30), r31 = bf16_add_rne(b31, l31);

        p0 = bf16_pack_halves(r00, r01);
        p1 = bf16_pack_halves(r10, r11);
        p2 = bf16_pack_halves(r20, r21);
        p3 = bf16_pack_halves(r30, r31);
    }
    else
    {
        p0 = fp16_pack_halves(m0[0], m0[1]);
        p1 = fp16_pack_halves(m1[0], m1[1]);
        p2 = fp16_pack_halves(m2[0], m2[1]);
        p3 = fp16_pack_halves(m3[0], m3[1]);
    }
}

CK_TILE_DEVICE void exchange_precomp(bool high_lane, int bperm_addr, uint32_t& lower, uint32_t& upper)
{
    const uint32_t src   = high_lane ? lower : upper;
    const uint32_t other = __builtin_amdgcn_ds_bpermute(bperm_addr, src);
    lower                = high_lane ? other : lower;
    upper                = high_lane ? upper : other;
}

CK_TILE_DEVICE Output4 permute_quad_to_row(bool high16,
                                           int addr16,
                                           bool high32,
                                           int addr32,
                                           uint32_t p0,
                                           uint32_t p1,
                                           uint32_t p2,
                                           uint32_t p3)
{
    exchange_precomp(high16, addr16, p0, p1);
    exchange_precomp(high16, addr16, p2, p3);
    exchange_precomp(high32, addr32, p0, p2);
    exchange_precomp(high32, addr32, p1, p3);
    return Output4{__builtin_amdgcn_perm(p1, p0, GatherLowHalves),
                   __builtin_amdgcn_perm(p3, p2, GatherLowHalves),
                   __builtin_amdgcn_perm(p1, p0, GatherHighHalves),
                   __builtin_amdgcn_perm(p3, p2, GatherHighHalves)};
}

CK_TILE_DEVICE void permute_two_quads_to_rows(bool high16,
                                              int addr16,
                                              bool high32,
                                              int addr32,
                                              uint32_t a0,
                                              uint32_t a1,
                                              uint32_t a2,
                                              uint32_t a3,
                                              uint32_t b0,
                                              uint32_t b1,
                                              uint32_t b2,
                                              uint32_t b3,
                                              Output4& outA,
                                              Output4& outB)
{
    const uint32_t oa0 = __builtin_amdgcn_ds_bpermute(addr16, high16 ? a0 : a1);
    const uint32_t oa1 = __builtin_amdgcn_ds_bpermute(addr16, high16 ? a2 : a3);
    const uint32_t ob0 = __builtin_amdgcn_ds_bpermute(addr16, high16 ? b0 : b1);
    const uint32_t ob1 = __builtin_amdgcn_ds_bpermute(addr16, high16 ? b2 : b3);

    a0 = high16 ? oa0 : a0;
    a1 = high16 ? a1 : oa0;
    a2 = high16 ? oa1 : a2;
    a3 = high16 ? a3 : oa1;
    const uint32_t oa2 = __builtin_amdgcn_ds_bpermute(addr32, high32 ? a0 : a2);
    const uint32_t oa3 = __builtin_amdgcn_ds_bpermute(addr32, high32 ? a1 : a3);

    b0 = high16 ? ob0 : b0;
    b1 = high16 ? b1 : ob0;
    b2 = high16 ? ob1 : b2;
    b3 = high16 ? b3 : ob1;
    const uint32_t ob2 = __builtin_amdgcn_ds_bpermute(addr32, high32 ? b0 : b2);
    const uint32_t ob3 = __builtin_amdgcn_ds_bpermute(addr32, high32 ? b1 : b3);

    a0   = high32 ? oa2 : a0;
    a2   = high32 ? a2 : oa2;
    a1   = high32 ? oa3 : a1;
    a3   = high32 ? a3 : oa3;
    outA = Output4{__builtin_amdgcn_perm(a1, a0, GatherLowHalves),
                   __builtin_amdgcn_perm(a3, a2, GatherLowHalves),
                   __builtin_amdgcn_perm(a1, a0, GatherHighHalves),
                   __builtin_amdgcn_perm(a3, a2, GatherHighHalves)};

    b0   = high32 ? ob2 : b0;
    b2   = high32 ? b2 : ob2;
    b1   = high32 ? ob3 : b1;
    b3   = high32 ? b3 : ob3;
    outB = Output4{__builtin_amdgcn_perm(b1, b0, GatherLowHalves),
                   __builtin_amdgcn_perm(b3, b2, GatherLowHalves),
                   __builtin_amdgcn_perm(b1, b0, GatherHighHalves),
                   __builtin_amdgcn_perm(b3, b2, GatherHighHalves)};
}

CK_TILE_DEVICE Rsrc4 make_e_rsrc(const void* ptr, uint32_t size_bytes)
{
    Rsrc4 r;
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    r[0] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(addr & 0xffffffffu));
    r[1] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(addr >> 32));
    r[2] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(size_bytes));
    r[3] = __builtin_amdgcn_readfirstlane(0x00020000);
    return r;
}

CK_TILE_DEVICE void buffer_store_b128(const Output4& out, uint32_t voff, Rsrc4 rsrc, uint32_t soff)
{
    asm volatile("buffer_store_dwordx4 %0, %1, %2, %3 offen"
                 :
                 : "v"(out), "v"(voff), "s"(rsrc), "s"(soff)
                 : "memory");
}
} // namespace int8_nt_paired_fast

template <class P, bool BFloat>
struct GroupedInt8NtScaledTileEpilogue
{
    using Acc    = typename P::Mmac::CVecType;
    using Policy = GroupedInt8NtTilePolicy<P>;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,
                                   index_t mb,
                                   index_t nb,
                                   int8_t* smem,
                                   const Acc (&c)[P::MR][P::NR],
                                   Int8NtScaleArgs scale) const
    {
        const index_t tid = get_thread_id(), lane = tid % P::WaveSize, wave = tid / P::WaveSize,
                      wm = wave / P::WN, wn = wave % P::WN;
        if constexpr(P::Aligned && (P::NR % 2 == 0))
        {
            const long long stride_bytes_ll =
                static_cast<long long>(a.stride_E) * static_cast<long long>(sizeof(uint16_t));
            const long long wave_span_bytes =
                static_cast<long long>((P::BM / P::WM) - 1) * stride_bytes_ll +
                static_cast<long long>(P::BN / P::WN) * static_cast<long long>(sizeof(uint16_t));
            if(scale.a_stride == 0 && (!scale.b || scale.b_stride == 1) && a.stride_E > 0 &&
               a.stride_E % 8 == 0 && reinterpret_cast<uintptr_t>(a.e_ptr) % 16 == 0 &&
               wave_span_bytes <= 0x7fffffffLL)
            {
                using namespace int8_nt_paired_fast;
                const uint32_t stride_bytes = static_cast<uint32_t>(stride_bytes_ll);
                const index_t wave_mb       = mb + wm * (P::BM / P::WM);
                const index_t wave_nb       = nb + wn * (P::BN / P::WN);
                auto* const wave_e_base     = static_cast<uint16_t*>(a.e_ptr) +
                                          static_cast<long long>(wave_mb) * a.stride_E + wave_nb;
                const Rsrc4 e_rsrc =
                    make_e_rsrc(wave_e_base, static_cast<uint32_t>(wave_span_bytes));
                const uint32_t e_voff = static_cast<uint32_t>(lane & 15) * stride_bytes +
                                        static_cast<uint32_t>(lane >> 4) * 16u;
                const bool high16        = (lane & 16) != 0;
                const bool high32        = (lane & 32) != 0;
                const int addr16         = (static_cast<int>(lane) ^ 16) * 4;
                const int addr32         = (static_cast<int>(lane) ^ 32) * 4;
                const float act_scale    = scale.a ? scale.a[0] : 1.0f;
                const FloatPair act_pair = {act_scale, act_scale};

                static_for<0, P::NR / 2, 1>{}([&](auto sp) {
                    constexpr index_t s = decltype(sp)::value;
                    FloatPair factors[4];
                    float left[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
                    float right[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                    if(scale.b)
                    {
                        const float* sb = scale.b + (wave_nb + s * 32 + (lane >> 4));
                        static_for<0, 4, 1>{}([&](auto pair) {
                            constexpr int p = decltype(pair)::value;
                            left[p]         = sb[p * 8];
                            right[p]        = sb[p * 8 + 4];
                        });
                    }
                    static_for<0, 4, 1>{}([&](auto pair) {
                        constexpr int p = decltype(pair)::value;
                        factors[p]      = pk_mul(act_pair, FloatPair{left[p], right[p]});
                        if constexpr(BFloat)
                        {
                            factors[p] = FloatPair{canon_factor(factors[p][0]),
                                                   canon_factor(factors[p][1])};
                        }
                    });
                    static_for<0, P::MR, 1>{}([&](auto mp) {
                        constexpr index_t mi = decltype(mp)::value;
                        uint32_t p0, p1, p2, p3;
                        scale_quad<BFloat>(
                            c[mi][s * 2], c[mi][s * 2 + 1], factors, p0, p1, p2, p3);
                        const Output4 out =
                            permute_quad_to_row(high16, addr16, high32, addr32, p0, p1, p2, p3);
                        const uint32_t soff = __builtin_amdgcn_readfirstlane(
                            static_cast<uint32_t>(mi * 16) * stride_bytes +
                            static_cast<uint32_t>(s * 64));
                        buffer_store_b128(out, e_voff, e_rsrc, soff);
                    });
                });
                return;
            }
        }
        auto* tmp = reinterpret_cast<int32_t*>(smem);
        const Int8NtDequantStore<BFloat> store{scale};
        if constexpr(Policy::OperandAllocation > 0)
            block_sync_lds();
        static_for<0, P::MR, 1>{}([&](auto i) {
            constexpr index_t mi = decltype(i)::value;
            if constexpr(P::BatchedEpi)
            {
                const index_t row = wm * P::MmacRows + lane % P::MmacRows;
                static_for<0, P::NR, 1>{}([&](auto j) {
                    static_for<0, P::AccElements, 1>{}([&](auto v) {
                        constexpr index_t ni = decltype(j)::value, vi = decltype(v)::value;
                        const index_t col = wn * (P::BN / P::WN) + ni * P::MmacCols +
                                            lane / P::MmacRows + vi * P::AccElements;
                        tmp[row * Policy::CPitch +
                            (col ^ ((row % Policy::CSwizzleRows) * Policy::CVectorElements))] =
                            c[mi][ni][vi];
                    });
                });
                block_sync_lds();
                for(index_t vec = tid; vec < Policy::CStripeRows * Policy::CVectorsPerRow;
                    vec += P::Threads)
                {
                    const index_t lds_row = vec / Policy::CVectorsPerRow;
                    const index_t col = (vec % Policy::CVectorsPerRow) * Policy::CVectorElements;
                    const index_t output_row = mb + (lds_row / P::MmacRows) * (P::BM / P::WM) +
                                               mi * P::MmacRows + lds_row % P::MmacRows;
                    if(P::Aligned ||
                       (output_row < a.M && nb + col + Policy::CVectorElements <= a.N))
                    {
                        const auto v = *reinterpret_cast<const typename P::CopyVector*>(
                            tmp + lds_row * Policy::CPitch +
                            (col ^ ((lds_row % Policy::CSwizzleRows) * Policy::CVectorElements)));
                        store.four(a, output_row, nb + col, v[0], v[1], v[2], v[3]);
                    }
                }
                if constexpr(mi + 1 < P::MR)
                    block_sync_lds();
            }
            else
            {
                static_for<0, P::WM, 1>{}([&](auto w) {
                    constexpr index_t wi = decltype(w)::value;
                    if(wm == wi)
                        static_for<0, P::NR, 1>{}([&](auto j) {
                            static_for<0, P::AccElements, 1>{}([&](auto v) {
                                constexpr index_t ni = decltype(j)::value, vi = decltype(v)::value;
                                const index_t col = wn * (P::BN / P::WN) + ni * P::MmacCols +
                                                    lane / P::MmacRows + vi * P::AccElements;
                                tmp[(lane % P::MmacRows) * Policy::CPitch + col] = c[mi][ni][vi];
                            });
                        });
                    block_sync_lds();
                    for(index_t vec = tid; vec < Policy::CStripeRows * Policy::CVectorsPerRow;
                        vec += P::Threads)
                    {
                        const index_t lds_row = vec / Policy::CVectorsPerRow;
                        const index_t col =
                            (vec % Policy::CVectorsPerRow) * Policy::CVectorElements;
                        const index_t output_row =
                            mb + wi * (P::BM / P::WM) + mi * P::MmacRows + lds_row;
                        if(P::Aligned ||
                           (output_row < a.M && nb + col + Policy::CVectorElements <= a.N))
                        {
                            const auto v = *reinterpret_cast<const typename P::CopyVector*>(
                                tmp + lds_row * Policy::CPitch + col);
                            store.four(a, output_row, nb + col, v[0], v[1], v[2], v[3]);
                        }
                    }
                    if constexpr(mi + 1 < P::MR || wi + 1 < P::WM)
                        block_sync_lds();
                });
            }
        });
    }
};
template <class P, bool BFloat>
struct GroupedInt8NtScaledWaspEpilogue
{
    using Tile = typename P::Pipeline::AccTile;
    // The native WASP domain has complete tiles (or complete 32-column consumer
    // stripes for BN384). Its output bounds are proved once in the kernel.
    // A scalar activation scale and channel/scalar weight scales are reused
    // over the accumulator tile rather than loaded for each output value.
    template <bool ChannelB>
    CK_TILE_DEVICE void scalar_a(const UniversalGemmKernelArgs<>& original,
                                 index_t mb,
                                 index_t nb,
                                 const Tile& tile,
                                 Int8NtScaleArgs scale,
                                 float sa,
                                 float sb) const
    {
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        if(wave < P::ProducerWaves)
            return;
        const index_t consumer = wave - P::ProducerWaves, h = lane / 16;
        if constexpr(P::BN == 384 || P::Pipeline::MasksInactiveN)
            if(nb + consumer * 32 >= (P::Transpose ? original.M : original.N))
                return;
        if constexpr(P::Transpose)
        {
            float factor[P::BM / 16];
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                if constexpr(P::Pipeline::MasksInactiveN)
                    if(mb + decltype(i)::value * 16 >= original.N)
                        return;
                const index_t col          = mb + decltype(i)::value * 16 + lane % 16;
                factor[decltype(i)::value] = sa * (ChannelB ? scale.b[col] : sb);
            });
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                if constexpr(P::Pipeline::MasksInactiveN)
                    if(mb + decltype(i)::value * 16 >= original.N)
                        return;
                const index_t col = mb + decltype(i)::value * 16 + lane % 16;
                static_for<0, 2, 1>{}([&](auto j) {
                    const auto v       = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                    const index_t base = nb + consumer * 32 + decltype(j)::value * 16;
                    static_for<0, 4, 1>{}([&](auto k) {
                        const index_t row = base + h + decltype(k)::value * 4;
                        static_cast<uint16_t*>(
                            original.e_ptr)[static_cast<long long>(row) * original.stride_E + col] =
                            Int8NtDequantConvert<BFloat>::run(v[decltype(k)::value],
                                                              factor[decltype(i)::value]);
                    });
                });
            });
        }
        else
        {
            float factor[2][4];
            static_for<0, 2, 1>{}([&](auto j) {
                static_for<0, 4, 1>{}([&](auto k) {
                    const index_t col =
                        nb + consumer * 32 + decltype(j)::value * 16 + h * 4 + decltype(k)::value;
                    factor[decltype(j)::value][decltype(k)::value] =
                        sa * (ChannelB ? scale.b[col] : sb);
                });
            });
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                const index_t row = mb + decltype(i)::value * 16 + lane % 16;
                static_for<0, 2, 1>{}([&](auto j) {
                    const auto v       = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                    const index_t base = nb + consumer * 32 + decltype(j)::value * 16;
                    int32_t r0 = v[0], r1 = v[1], r2 = v[2], r3 = v[3];
                    transpose_mmac_4x4(lane, h, r0, r1, r2, r3);
                    using V       = uint16_t __attribute__((ext_vector_type(4)));
                    const auto* f = factor[decltype(j)::value];
                    const V out{Int8NtDequantConvert<BFloat>::run(r0, f[0]),
                                Int8NtDequantConvert<BFloat>::run(r1, f[1]),
                                Int8NtDequantConvert<BFloat>::run(r2, f[2]),
                                Int8NtDequantConvert<BFloat>::run(r3, f[3])};
                    *reinterpret_cast<V*>(static_cast<uint16_t*>(original.e_ptr) +
                                          static_cast<long long>(row) * original.stride_E + base +
                                          h * 4) = out;
                });
            });
        }
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& original,
                                   index_t mb,
                                   index_t nb,
                                   const Tile& tile,
                                   Int8NtScaleArgs scale) const
    {
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        if(wave < P::ProducerWaves)
            return;
        if(scale.a_stride == 0)
        {
            const float sa = scale.a ? scale.a[0] : 1.0f;
            if(scale.b && scale.b_stride == 1)
                scalar_a<true>(original, mb, nb, tile, scale, sa, 1.0f);
            else
            {
                const float sb = scale.b ? scale.b[0] : 1.0f;
                scalar_a<false>(original, mb, nb, tile, scale, sa, sb);
            }
            return;
        }
        const index_t consumer = wave - P::ProducerWaves, h = lane / 16;
        const Int8NtDequantStore<BFloat> store{scale};
        static_for<0, P::BM / 16, 1>{}([&](auto i) {
            const index_t row = mb + decltype(i)::value * 16 + lane % 16;
            static_for<0, 2, 1>{}([&](auto j) {
                const auto v       = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                const index_t base = nb + consumer * 32 + decltype(j)::value * 16;
                if constexpr(P::Transpose)
                {
                    static_for<0, 4, 1>{}([&](auto k) {
                        const index_t col = base + h + decltype(k)::value * 4;
                        if(col < original.M && row < original.N)
                            store.one(original, col, row, v[decltype(k)::value]);
                    });
                }
                else
                {
                    int32_t r0 = v[0], r1 = v[1], r2 = v[2], r3 = v[3];
                    transpose_mmac_4x4(lane, h, r0, r1, r2, r3);
                    if(row < original.M && base + h * 4 + 4 <= original.N)
                        store.four(original, row, base + h * 4, r0, r1, r2, r3);
                }
            });
        });
    }
};

// -----------------------------------------------------------------------------
// Paired Store Epilogue: paired FP32 arithmetic, then packed BF16/FP16 stores.
// -----------------------------------------------------------------------------
namespace int8_nt_paired_store {
using FloatPair = ext_vector_t<float, 2>;
// v_perm selectors choose the high 16 bits of each rounded FP32 bit pattern.
constexpr uint32_t PackRoundedBf16Halves = 0x07060302U;
// The following selectors gather two low or two high 16-bit output halves.
constexpr uint32_t GatherLowHalves  = 0x05040100U;
constexpr uint32_t GatherHighHalves = 0x07060302U;

CK_TILE_DEVICE FloatPair multiply(FloatPair values, FloatPair factors)
{
    FloatPair result;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(values), "v"(factors));
    return result;
}

CK_TILE_DEVICE float canonicalize_factor(float f)
{
    const uint32_t bits = bit_cast<uint32_t>(f);
    const uint32_t nan =
        (bits & int8_nt_bf16_bits::Fp32HighHalfMask) | int8_nt_bf16_bits::Fp32QuietNanBit;
    uint32_t out;
    asm volatile("v_cmp_u_f32 vcc, %1, %1\n\t"
                 "v_cndmask_b32 %0, %1, %2, vcc"
                 : "=v"(out)
                 : "v"(f), "v"(nan)
                 : "vcc");
    return bit_cast<float>(out);
}

CK_TILE_DEVICE uint32_t round_bf16_bits(float result)
{
    const uint32_t bits = bit_cast<uint32_t>(result);
    uint32_t lsb, rounded;
    asm volatile("v_bfe_u32 %[lsb], %[bits], 16, 1\n\t"
                 "v_add3_u32 %[rounded], %[bits], %[bias], %[lsb]"
                 : [lsb] "=&v"(lsb), [rounded] "=&v"(rounded)
                 : [bits] "v"(bits), [bias] "s"(int8_nt_bf16_bits::RoundToNearestEvenBias));
    return rounded;
}

template <bool BFloat>
CK_TILE_DEVICE uint16_t convert(float result)
{
    if constexpr(BFloat)
    {
        const uint32_t bits    = bit_cast<uint32_t>(result);
        const uint32_t rounded = round_bf16_bits(result) >> 16;
        const uint32_t nan =
            (bits >> int8_nt_bf16_bits::DiscardedBits) | int8_nt_bf16_bits::Bf16QuietNanBit;
        uint32_t selected;
        asm volatile("v_cmp_u_f32 vcc, %1, %1\n\t"
                     "v_cndmask_b32 %0, %2, %3, vcc"
                     : "=v"(selected)
                     : "v"(result), "v"(rounded), "v"(nan)
                     : "vcc");
        return static_cast<uint16_t>(selected);
    }
    else
        return bit_cast<uint16_t>(type_convert<half_t>(result));
}

template <bool BFloat>
CK_TILE_DEVICE uint32_t scale_pair(int32_t first, int32_t second, FloatPair factors)
{
    const FloatPair values{static_cast<float>(first), static_cast<float>(second)};
    const FloatPair result = multiply(values, factors);
    if constexpr(BFloat)
    {
        const uint32_t r0 = round_bf16_bits(result[0]);
        const uint32_t r1 = round_bf16_bits(result[1]);
        uint32_t packed;
        asm volatile("v_perm_b32 %0, %1, %2, %3"
                     : "=v"(packed)
                     : "v"(r1), "v"(r0), "s"(PackRoundedBf16Halves));
        return packed;
    }
    else
    {
        const uint16_t h0 = convert<false>(result[0]);
        const uint16_t h1 = convert<false>(result[1]);
        uint32_t packed;
        asm volatile("v_pack_b32_f16 %0, %1, %2"
                     : "=v"(packed)
                     : "v"(h0), "v"(h1));
        return packed;
    }
}

// Swap one lane bit and one register index bit. Both halves of each packed
// pair travel together. Every lane participates before the store.
template <int Mask>
CK_TILE_DEVICE void exchange(int lane, uint32_t& lower, uint32_t& upper)
{
    const bool high_lane = (lane & Mask) != 0;
    const auto other = __builtin_amdgcn_ds_bpermute((lane ^ Mask) * 4, high_lane ? lower : upper);
    lower            = high_lane ? other : lower;
    upper            = high_lane ? upper : other;
}
} // namespace int8_nt_paired_store

template <class Problem, class = void>
struct Int8NtPairedBlocksDefault
{
    static constexpr bool value = false;
};

template <class Problem>
struct Int8NtPairedBlocksDefault<Problem, std::void_t<decltype(Problem::PairEpi)>>
{
    static constexpr bool value = Problem::PairEpi;
};

template <class Problem, class = void>
struct Int8NtLdsEpiTrait
{
    static constexpr bool value = false;
};

template <class Problem>
struct Int8NtLdsEpiTrait<Problem, std::void_t<decltype(Problem::LdsEpi)>>
{
    static constexpr bool value = Problem::LdsEpi;
};

template <class Problem, bool BFloat, bool PairBlocks = Int8NtPairedBlocksDefault<Problem>::value>
struct GroupedInt8NtPairedStoreEpilogue
{
    using Tile = typename Problem::Pipeline::AccTile;

    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   const Tile& tile,
                                   Int8NtScaleArgs scale,
                                   int8_t* smem = nullptr) const
    {
        using namespace int8_nt_paired_store;
        static_assert(!Problem::Transpose);
        const int tid      = get_thread_id();
        const int lane     = tid % 64;
        const int wave     = tid / 64;
        const int consumer = wave - Problem::ProducerWaves;
        const int stripe   = nb + consumer * 32;
        if constexpr(!Int8NtLdsEpiTrait<Problem>::value)
        {
            if(consumer < 0 || stripe >= arg.N)
                return;
        }

        const bool high16               = (lane & 16) != 0;
        const bool high32               = (lane & 32) != 0;
        const int addr16                = (lane ^ 16) * 4;
        const int addr32                = (lane ^ 32) * 4;
        const long long stride_bytes_ll =
            static_cast<long long>(arg.stride_E) * static_cast<long long>(sizeof(uint16_t));
        const long long span_bytes_ll =
            static_cast<long long>(Problem::BM - 1) * stride_bytes_ll + 64LL;

        // Single non-short-circuit predicate keeps LLVM from placing the cold fallback before fast_ok.
        const uint32_t bad_fast =
            static_cast<uint32_t>(scale.a_stride != 0) |
            static_cast<uint32_t>(scale.b != nullptr && scale.b_stride > 1) |
            static_cast<uint32_t>(arg.stride_E <= 0) |
            static_cast<uint32_t>((arg.stride_E & 7) != 0) |
            static_cast<uint32_t>((reinterpret_cast<uintptr_t>(arg.e_ptr) & 15u) != 0) |
            static_cast<uint32_t>(span_bytes_ll > 0x7fffffffLL);

        if(__builtin_expect(bad_fast == 0u, 1))
        {
            float left[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
            float right[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            if(scale.b && (!Int8NtLdsEpiTrait<Problem>::value || (consumer >= 0 && stripe < arg.N)))
            {
                const float* sb     = scale.b + (stripe + (lane >> 4)) * scale.b_stride;
                const index_t sstep = scale.b_stride;
                static_for<0, 4, 1>{}([&](auto pair) {
                    constexpr int p = decltype(pair)::value;
                    left[p]         = sb[(p * 8) * sstep];
                    right[p]        = sb[(p * 8 + 4) * sstep];
                });
            }
            const float activation_scale = scale.a ? scale.a[0] : 1.0f;
            FloatPair factors[4];
            static_for<0, 4, 1>{}([&](auto pair) {
                constexpr int p = decltype(pair)::value;
                factors[p]      = multiply(FloatPair{activation_scale, activation_scale},
                                      FloatPair{left[p], right[p]});
                if constexpr(BFloat)
                {
                    factors[p] = FloatPair{canonicalize_factor(factors[p][0]),
                                           canonicalize_factor(factors[p][1])};
                }
            });
            const uint32_t stride_bytes = static_cast<uint32_t>(stride_bytes_ll);
            auto* const e_base          = static_cast<uint16_t*>(arg.e_ptr) +
                                 static_cast<long long>(mb) * arg.stride_E + stripe;
            const auto e_rsrc =
                int8_nt_paired_fast::make_e_rsrc(e_base, static_cast<uint32_t>(span_bytes_ll));
            const uint32_t e_voff = static_cast<uint32_t>(lane & 15) * stride_bytes +
                                    static_cast<uint32_t>(lane >> 4) * 16u;
            if constexpr(Int8NtLdsEpiTrait<Problem>::value)
            {
                // 8-pass double-buffered 32-row LDS transpose across all 16 waves:
                // 784 B row stride avoids 8-lane LDS bank conflicts while fitting 2x32 rows in 64 KB.
                // Align-shifted read/store lanes keep 4-lane and 16-lane global stores within 64B lines.
                static_assert(Problem::BM == 256 && Problem::BN == 384 &&
                              Problem::ProducerWaves + Problem::Pipeline::kNumConsumerWaves == 16);
                constexpr int kLdsRowBytes = 784;
                constexpr int kBufBytes    = 32 * kLdsRowBytes;
                const int valid_m          = min(index_t(Problem::BM), arg.M - mb);
                const int valid_n          = min(index_t(Problem::BN), arg.N - nb);
                const long long row_span_ll =
                    static_cast<long long>(valid_m - 1) * stride_bytes_ll +
                    static_cast<long long>(valid_n) * static_cast<long long>(sizeof(uint16_t));
                auto* const row_e_base = static_cast<uint16_t*>(arg.e_ptr) +
                                         static_cast<long long>(mb) * arg.stride_E + nb;
                const auto row_rsrc = int8_nt_paired_fast::make_e_rsrc(
                    row_e_base, static_cast<uint32_t>(row_span_ll));
                const bool do_write   = (consumer >= 0 && stripe < arg.N);
                const int lds_col_off = do_write ? ((consumer * 4 + (lane >> 4)) * 16) : 0;
                const int lds_w_base  = (lane & 15) * kLdsRowBytes + lds_col_off;
                const int align_shift = static_cast<int>(
                    ((reinterpret_cast<uintptr_t>(row_e_base) +
                      static_cast<uintptr_t>(wave) * static_cast<uintptr_t>(stride_bytes_ll)) &
                     63u) >>
                    4);
                const int slot        = lane - align_shift;
                const bool do_read    = (slot >= 0 && slot < 48 && slot * 8 < valid_n);
                const int lds_r_base  = wave * kLdsRowBytes + (do_read ? (slot * 16) : 0);
                const uint32_t voff_r = do_read ? (static_cast<uint32_t>(wave) * stride_bytes +
                                                   static_cast<uint32_t>(slot * 16))
                                                : 0x7fffffffu;
                auto write_pair = [&](int blk_base, int lds_off) __attribute__((always_inline)) {
                    uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                    int8_nt_paired_fast::scale_quad<BFloat>(
                        tile.v[blk_base * 2], tile.v[blk_base * 2 + 1], factors, a0, a1, a2, a3);
                    int8_nt_paired_fast::scale_quad<BFloat>(
                        tile.v[(blk_base + 1) * 2], tile.v[(blk_base + 1) * 2 + 1], factors, b0, b1, b2, b3);
                    int8_nt_paired_fast::Output4 outA, outB;
                    int8_nt_paired_fast::permute_two_quads_to_rows(
                        high16, addr16, high32, addr32, a0, a1, a2, a3, b0, b1, b2, b3, outA, outB);
                    *reinterpret_cast<int8_nt_paired_fast::Output4*>(smem + lds_off + lds_w_base) =
                        outA;
                    *reinterpret_cast<int8_nt_paired_fast::Output4*>(
                        smem + lds_off + 16 * kLdsRowBytes + lds_w_base) = outB;
                };
                if(__builtin_expect(do_write, 1))
                    write_pair(0, 0);
                __syncthreads();
                static_for<0, 8, 1>{}([&](auto pp) {
                    constexpr int pass    = decltype(pp)::value;
                    constexpr int cur_off = (pass & 1) * kBufBytes;
                    const auto r0 = *reinterpret_cast<const int8_nt_paired_fast::Output4*>(
                        smem + cur_off + lds_r_base);
                    const auto r1 = *reinterpret_cast<const int8_nt_paired_fast::Output4*>(
                        smem + cur_off + 16 * kLdsRowBytes + lds_r_base);
                    if constexpr(pass + 1 < 8)
                    {
                        constexpr int nxt_off = ((pass + 1) & 1) * kBufBytes;
                        if(__builtin_expect(do_write, 1))
                            write_pair((pass + 1) * 2, nxt_off);
                    }
                    const uint32_t soff0 = __builtin_amdgcn_readfirstlane(
                        static_cast<uint32_t>(pass * 32) * stride_bytes);
                    const uint32_t soff1 = __builtin_amdgcn_readfirstlane(
                        static_cast<uint32_t>(pass * 32 + 16) * stride_bytes);
                    int8_nt_paired_fast::buffer_store_b128(r0, voff_r, row_rsrc, soff0);
                    int8_nt_paired_fast::buffer_store_b128(r1, voff_r, row_rsrc, soff1);
                    if constexpr(pass + 1 < 8)
                        __syncthreads();
                });
                return;
            }
            if constexpr(PairBlocks && (Problem::BM / 16) % 2 == 0)
            {
                static_for<0, Problem::BM / 32, 1>{}([&](auto bp) {
                    constexpr int b0i = decltype(bp)::value * 2;
                    constexpr int b1i = b0i + 1;
                    uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                    int8_nt_paired_fast::scale_quad<BFloat>(
                        tile.v[b0i * 2], tile.v[b0i * 2 + 1], factors, a0, a1, a2, a3);
                    int8_nt_paired_fast::scale_quad<BFloat>(
                        tile.v[b1i * 2], tile.v[b1i * 2 + 1], factors, b0, b1, b2, b3);
                    int8_nt_paired_fast::Output4 outA, outB;
                    int8_nt_paired_fast::permute_two_quads_to_rows(
                        high16, addr16, high32, addr32, a0, a1, a2, a3, b0, b1, b2, b3, outA, outB);
                    const uint32_t soffA = __builtin_amdgcn_readfirstlane(
                        static_cast<uint32_t>(b0i * 16) * stride_bytes);
                    const uint32_t soffB = __builtin_amdgcn_readfirstlane(
                        static_cast<uint32_t>(b1i * 16) * stride_bytes);
                    int8_nt_paired_fast::buffer_store_b128(outA, e_voff, e_rsrc, soffA);
                    int8_nt_paired_fast::buffer_store_b128(outB, e_voff, e_rsrc, soffB);
                });
            }
            else
            {
                static_for<0, Problem::BM / 16, 1>{}([&](auto block) {
                    constexpr int i = decltype(block)::value;
                    uint32_t p0, p1, p2, p3;
                    int8_nt_paired_fast::scale_quad<BFloat>(
                        tile.v[i * 2], tile.v[i * 2 + 1], factors, p0, p1, p2, p3);
                    const auto out = int8_nt_paired_fast::permute_quad_to_row(
                        high16, addr16, high32, addr32, p0, p1, p2, p3);
                    const uint32_t soff =
                        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(i * 16) * stride_bytes);
                    int8_nt_paired_fast::buffer_store_b128(out, e_voff, e_rsrc, soff);
                });
            }
            return;
        }

        if constexpr(Int8NtLdsEpiTrait<Problem>::value)
        {
            if(consumer < 0 || stripe >= arg.N)
                return;
        }
        // Compact cold fallback keeps entire kernel under 32 KB SQC I-cache.
        float b_left[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
        float b_right[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        if(scale.b)
        {
            const index_t b_col_base = stripe + (lane >> 4);
            static_for<0, 4, 1>{}([&](auto pair) {
                constexpr int p = decltype(pair)::value;
                b_left[p]       = scale.b[(b_col_base + p * 8) * scale.b_stride];
                b_right[p]      = scale.b[(b_col_base + p * 8 + 4) * scale.b_stride];
            });
        }
        const int col = stripe + (lane >> 4) * 8;
        static_for<0, Problem::BM / 16, 1>{}([&](auto block) {
            constexpr int i   = decltype(block)::value;
            const int row     = mb + i * 16 + (lane & 15);
            const float act   = scale.a ? scale.a[scale.a_stride ? row * scale.a_stride : 0] : 1.0f;
            FloatPair factors[4];
            static_for<0, 4, 1>{}([&](auto pair) {
                constexpr int p = decltype(pair)::value;
                factors[p]      = multiply(FloatPair{act, act}, FloatPair{b_left[p], b_right[p]});
                if constexpr(BFloat)
                {
                    factors[p] = FloatPair{canonicalize_factor(factors[p][0]),
                                           canonicalize_factor(factors[p][1])};
                }
            });
            uint32_t p0, p1, p2, p3;
            int8_nt_paired_fast::scale_quad<BFloat>(
                tile.v[i * 2], tile.v[i * 2 + 1], factors, p0, p1, p2, p3);
            const auto out = int8_nt_paired_fast::permute_quad_to_row(
                high16, addr16, high32, addr32, p0, p1, p2, p3);
            if(row < arg.M)
            {
                auto* dst = static_cast<uint16_t*>(arg.e_ptr) +
                            static_cast<long long>(row) * arg.stride_E + col;
                static_for<0, 4, 1>{}([&](auto w) {
                    constexpr int wi    = decltype(w)::value;
                    const uint32_t pair = out[wi];
                    if(col + wi * 2 < arg.N)
                        dst[wi * 2] = static_cast<uint16_t>(pair & 0xffffu);
                    if(col + wi * 2 + 1 < arg.N)
                        dst[wi * 2 + 1] = static_cast<uint16_t>(pair >> 16);
                });
            }
        });
    }
};
} // namespace ck_tile

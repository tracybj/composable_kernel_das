// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_shared.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_shared_tuned.hpp"
#include <cstdlib>

namespace ck_tile {
template<int Rows, int BlockK, int Depth>
struct DecodeNnSharedKernel
{
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* args,
                                   int groups, int tiles_per_group) const
    {
        const int block = get_block_1d_id();
        const int group = block / tiles_per_group;
        if(group >= groups) return;
        const auto* descriptors = reinterpret_cast<const GemmTransKernelArgImpl<0>*>(
            cast_pointer_to_generic_address_space(args));
        const auto& a = descriptors[group].group_karg;
        decode_nn_mainline::Params p{
            reinterpret_cast<const uint16_t*>(a.as_ptr[0]),
            reinterpret_cast<const uint16_t*>(a.bs_ptr[0]),
            reinterpret_cast<uint16_t*>(a.e_ptr),
            a.M, a.N, a.K, a.stride_As[0], a.stride_Bs[0], a.stride_E,
            (block - group * tiles_per_group) * 128};
        decode_nn_shared::run<Rows, BlockK, Depth>(p);
    }
};

template<int Rows, int BlockK, int Depth>
CK_TILE_HOST int launch_decode_shared(void* args, int groups, hipStream_t stream,
                                      uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_shared::Geometry<Rows, BlockK, Depth>;
    const auto kernel = make_kernel<256, 2>(DecodeNnSharedKernel<Rows, BlockK, Depth>{},
        dim3(static_cast<unsigned>(tiles)), dim3(256), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
CK_TILE_HOST int dispatch_decode_shared(void* args, int groups, hipStream_t stream,
                                        uint64_t tiles, int tiles_per_group, int variant)
{
    switch(variant)
    {
    case 0: case 3: return launch_decode_shared<Rows, 16, 4>(args, groups, stream, tiles, tiles_per_group);
    case 1: return launch_decode_shared<Rows, 16, 8>(args, groups, stream, tiles, tiles_per_group);
    case 2: return launch_decode_shared<Rows, 32, 4>(args, groups, stream, tiles, tiles_per_group);
    default: return -1;
    }
}
template<int Rows, int BlockK, int Depth, bool EarlyB, bool XorA>
struct DecodeNnSharedTunedKernel
{
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* args,
                                   int groups, int tiles_per_group) const
    {
        const int block = get_block_1d_id();
        const int group = block / tiles_per_group;
        if(group >= groups) return;
        const auto* descriptors = reinterpret_cast<const GemmTransKernelArgImpl<0>*>(
            cast_pointer_to_generic_address_space(args));
        const auto& a = descriptors[group].group_karg;
        decode_nn_mainline::Params p{
            reinterpret_cast<const uint16_t*>(a.as_ptr[0]),
            reinterpret_cast<const uint16_t*>(a.bs_ptr[0]),
            reinterpret_cast<uint16_t*>(a.e_ptr),
            a.M, a.N, a.K, a.stride_As[0], a.stride_Bs[0], a.stride_E,
            (block - group * tiles_per_group) * 128};
        decode_nn_shared_tuned::run<Rows, BlockK, Depth, EarlyB, XorA>(p);
    }
};

template<int Rows, int BlockK, int Depth, bool EarlyB, bool XorA>
CK_TILE_HOST int launch_decode_shared_tuned(void* args, int groups, hipStream_t stream,
                                            uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_shared::Geometry<Rows, BlockK, Depth>;
    const auto kernel = make_kernel<256, 2>(
        DecodeNnSharedTunedKernel<Rows, BlockK, Depth, EarlyB, XorA>{},
        dim3(static_cast<unsigned>(tiles)), dim3(256), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows, bool EarlyB, bool XorA>
CK_TILE_HOST int dispatch_decode_shared_tuned(void* args, int groups, hipStream_t stream,
                                              uint64_t tiles, int tiles_per_group, int variant)
{
    switch(variant)
    {
    case 0: case 3: return launch_decode_shared_tuned<Rows, 16, 4, EarlyB, XorA>(args, groups, stream, tiles, tiles_per_group);
    case 1: return launch_decode_shared_tuned<Rows, 16, 8, EarlyB, XorA>(args, groups, stream, tiles, tiles_per_group);
    case 2: return launch_decode_shared_tuned<Rows, 32, 4, EarlyB, XorA>(args, groups, stream, tiles, tiles_per_group);
    default: return -1;
    }
}

CK_TILE_HOST bool decode_shared_flag(const char* enable, const char* disable)
{
    const char* e = std::getenv(enable);
    const char* d = std::getenv(disable);
    return e != nullptr && e[0] == '1' && e[1] == '\0' &&
        !(d != nullptr && d[0] == '1' && d[1] == '\0');
}

template<int Rows>
CK_TILE_HOST int dispatch_decode_shared_flags(void* args, int groups, hipStream_t stream,
                                              uint64_t tiles, int tiles_per_group, int variant,
                                              bool early, bool xor_a)
{
    if(early && xor_a)
        return dispatch_decode_shared_tuned<Rows, true, true>(args, groups, stream, tiles, tiles_per_group, variant);
    if(early)
        return dispatch_decode_shared_tuned<Rows, true, false>(args, groups, stream, tiles, tiles_per_group, variant);
    if(xor_a)
        return dispatch_decode_shared_tuned<Rows, false, true>(args, groups, stream, tiles, tiles_per_group, variant);
    return dispatch_decode_shared<Rows>(args, groups, stream, tiles, tiles_per_group, variant);
}

} // namespace ck_tile

extern "C" int ck_tile_hcu_grouped_gemm_bf16_nn_decode_shared_device_args(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group,
    int m, int n, int k, int variant)
{
    if(args == nullptr || groups < 1 || groups > 1024 || m < 1 || m > 64 ||
       variant < 0 || variant > 3 || n % 128 != 0 || n > 4096) return -1;
    const bool short_path = variant == 3;
    if(n < (short_path ? 128 : 1024) ||
       (!short_path && (k < 2048 || k > 8192 || k % 128 != 0)) ||
       (short_path && (groups > 32 || k < 64 || k >= 2048 || k % 64 != 0)) ||
       tiles_per_group != n / 128 ||
       tiles != static_cast<uint64_t>(groups) * tiles_per_group) return -1;
    const bool early = ck_tile::decode_shared_flag(
        "CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_SHARED_EARLY_B",
        "CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_SHARED_EARLY_B");
    const bool xor_a = ck_tile::decode_shared_flag(
        "CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_SHARED_XOR_A",
        "CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_SHARED_XOR_A");
    if(m <= 16) return ck_tile::dispatch_decode_shared_flags<1>(args, groups, stream, tiles, tiles_per_group, variant, early, xor_a);
    if(m <= 32) return ck_tile::dispatch_decode_shared_flags<2>(args, groups, stream, tiles, tiles_per_group, variant, early, xor_a);
    return ck_tile::dispatch_decode_shared_flags<4>(args, groups, stream, tiles, tiles_per_group, variant, early, xor_a);
}

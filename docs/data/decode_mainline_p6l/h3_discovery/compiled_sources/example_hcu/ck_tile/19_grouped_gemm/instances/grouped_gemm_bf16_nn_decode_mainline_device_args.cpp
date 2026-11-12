// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_mainline.hpp"

namespace ck_tile {
template<int Rows, int Depth, int Waves, bool OperandLookahead>
struct DecodeNnMainlineKernel
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
        const int tile = block - group * tiles_per_group;
        decode_nn_mainline::Params p{
            reinterpret_cast<const uint16_t*>(a.as_ptr[0]),
            reinterpret_cast<const uint16_t*>(a.bs_ptr[0]),
            reinterpret_cast<uint16_t*>(a.e_ptr),
            a.M, a.N, a.K, a.stride_As[0], a.stride_Bs[0], a.stride_E,
            tile * Waves * 32};
        decode_nn_mainline::run<Rows, Depth, OperandLookahead>(p);
    }
};

template<int Rows, int Depth, int Waves, bool OperandLookahead>
CK_TILE_HOST int launch_decode_mainline(void* args, int groups, hipStream_t stream,
                                        uint64_t tiles, int tiles_per_group)
{
    const auto kernel = make_kernel<Waves * 64, 2>(
        DecodeNnMainlineKernel<Rows, Depth, Waves, OperandLookahead>{},
        dim3(static_cast<unsigned>(tiles)), dim3(Waves * 64), Depth * Waves * 1024,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
CK_TILE_HOST int dispatch_decode_mainline(void* args, int groups, hipStream_t stream,
                                          uint64_t tiles, int tiles_per_group, int variant)
{
    switch(variant)
    {
    case 0: return launch_decode_mainline<Rows, 4, 4, true>(args, groups, stream, tiles, tiles_per_group);
    case 1: return launch_decode_mainline<Rows, 8, 4, true>(args, groups, stream, tiles, tiles_per_group);
    case 2: return launch_decode_mainline<Rows, 4, 4, false>(args, groups, stream, tiles, tiles_per_group);
    case 3: return launch_decode_mainline<Rows, 2, 2, false>(args, groups, stream, tiles, tiles_per_group);
    case 4: return launch_decode_mainline<Rows, 2, 1, false>(args, groups, stream, tiles, tiles_per_group);
    default: return -1;
    }
}
} // namespace ck_tile

extern "C" int ck_tile_hcu_grouped_gemm_bf16_nn_decode_mainline_device_args(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group,
    int m, int n, int k, int variant)
{
    if(args == nullptr || groups < 1 || groups > 1024 || m < 1 || m > 64 ||
       variant < 0 || variant > 4) return -1;
    const int tile_n = variant == 3 ? 64 : variant == 4 ? 32 : 128;
    const bool mainline = variant < 3;
    if(n < (mainline ? 1024 : 128) || n > 4096 || n % 128 != 0 ||
       (mainline && (k < 2048 || k > 8192 || k % 128 != 0)) ||
       (!mainline && (groups > 32 || k < 64 || k >= 2048 || k % 32 != 0)) ||
       tiles_per_group != n / tile_n ||
       tiles != static_cast<uint64_t>(groups) * tiles_per_group) return -1;
    if(m <= 16) return ck_tile::dispatch_decode_mainline<1>(args, groups, stream, tiles, tiles_per_group, variant);
    if(m <= 32) return ck_tile::dispatch_decode_mainline<2>(args, groups, stream, tiles, tiles_per_group, variant);
    return ck_tile::dispatch_decode_mainline<4>(args, groups, stream, tiles, tiles_per_group, variant);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_wave.hpp"

namespace ck_tile {
template<int Depth>
struct DecodeNnWaveKernel
{
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* args,
                                   int groups, int tiles_per_group) const
    {
        const int block = ck_tile::get_block_1d_id();
        const int group = block / tiles_per_group;
        if(group >= groups) return;
        const auto* desc = reinterpret_cast<const ck_tile::GemmTransKernelArgImpl<0>*>(
            ck_tile::cast_pointer_to_generic_address_space(args));
        const auto& a = desc[group].group_karg;
        const int tile = block - group * tiles_per_group;
        const int ntiles = a.N / 128;
        ck_tile::decode_nn_wave::Params p{
            reinterpret_cast<const uint16_t*>(a.as_ptr[0]),
            reinterpret_cast<const uint16_t*>(a.bs_ptr[0]),
            reinterpret_cast<uint16_t*>(a.e_ptr),
            a.M, a.N, a.K, a.stride_As[0], a.stride_Bs[0], a.stride_E,
            (tile / ntiles) * 16, (tile % ntiles) * 128};
        ck_tile::decode_nn_wave::run<Depth>(p);
    }
};
template<int Depth>
CK_TILE_HOST int launch_decode_nn(void* args, int groups, hipStream_t stream,
                                  uint64_t tiles, int tiles_per_group)
{
    const auto kernel = make_kernel<256, 2>(
        DecodeNnWaveKernel<Depth>{}, dim3(static_cast<unsigned>(tiles)), dim3(256),
        Depth * 4096, cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}
} // namespace ck_tile

extern "C" int ck_tile_hcu_grouped_gemm_bf16_nn_decode_wave_device_args(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group, int k)
{
    if(args == nullptr || groups <= 0 || groups > 1024 || tiles == 0 ||
       tiles > (uint64_t{1} << 30) || tiles_per_group <= 0)
        return -1;
    return k >= 128 && k % 128 == 0
        ? ck_tile::launch_decode_nn<8>(args, groups, stream, tiles, tiles_per_group)
        : ck_tile::launch_decode_nn<2>(args, groups, stream, tiles, tiles_per_group);
}

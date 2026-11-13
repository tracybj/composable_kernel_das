// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_compact.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_wide.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_resource_phase.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_k64.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_k64_wide.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_n256.hpp"

namespace ck_tile {
template<int Rows, int BlockK, int Depth>
struct DecodeNnCooperativeKernel
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
        decode_nn_cooperative::run<Rows, BlockK, Depth>(p);
    }
};

template<int Rows, int BlockK, int Depth>
CK_TILE_HOST int launch_decode_cooperative(void* args, int groups, hipStream_t stream,
                                           uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative::Geometry<Rows, BlockK, Depth>;
    const auto kernel = make_kernel<256, 2>(DecodeNnCooperativeKernel<Rows, BlockK, Depth>{},
        dim3(static_cast<unsigned>(tiles)), dim3(256), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
struct DecodeNnCooperativeCompactKernel
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
        decode_nn_cooperative_compact::run<Rows>(p);
    }
};

template<int Rows>
CK_TILE_HOST int launch_decode_cooperative_compact(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_compact::Geometry<Rows>;
    const auto kernel = make_kernel<256, 2>(DecodeNnCooperativeCompactKernel<Rows>{},
        dim3(static_cast<unsigned>(tiles)), dim3(256), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
struct DecodeNnCooperativeWideKernel
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
        decode_nn_cooperative_wide::run<Rows, 32, 4>(p);
    }
};

template<int Rows>
CK_TILE_HOST int launch_decode_cooperative_wide(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_wide::Geometry<Rows, 32, 4>;
    const auto kernel = make_kernel<512, 1>(DecodeNnCooperativeWideKernel<Rows>{},
        dim3(static_cast<unsigned>(tiles)), dim3(512), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
struct DecodeNnCooperativeCompactPhaseKernel
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
        decode_nn_resource_phase::run_compact<Rows>(p);
    }
};

template<int Rows>
CK_TILE_HOST int launch_decode_cooperative_compact_phase(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_compact::Geometry<Rows>;
    const auto kernel = make_kernel<256, 2>(DecodeNnCooperativeCompactPhaseKernel<Rows>{},
        dim3(static_cast<unsigned>(tiles)), dim3(256), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
struct DecodeNnCooperativeWidePhaseKernel
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
        decode_nn_resource_phase::run_wide<Rows>(p);
    }
};

template<int Rows>
CK_TILE_HOST int launch_decode_cooperative_wide_phase(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_wide::Geometry<Rows, 32, 4>;
    const auto kernel = make_kernel<512, 1>(DecodeNnCooperativeWidePhaseKernel<Rows>{},
        dim3(static_cast<unsigned>(tiles)), dim3(512), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
struct DecodeNnCooperativeK64Kernel
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
        decode_nn_cooperative_k64::run<Rows>(p);
    }
};

template<int Rows>
CK_TILE_HOST int launch_decode_cooperative_k64(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_k64::Geometry<Rows,64,2>;
    const auto kernel = make_kernel<256, 2>(DecodeNnCooperativeK64Kernel<Rows>{},
        dim3(static_cast<unsigned>(tiles)), dim3(256), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
struct DecodeNnCooperativeK64WideKernel
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
        decode_nn_cooperative_k64_wide::run<Rows>(p);
    }
};

template<int Rows>
CK_TILE_HOST int launch_decode_cooperative_k64_wide(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_k64_wide::Geometry<Rows,64,2>;
    const auto kernel = make_kernel<512, 1>(DecodeNnCooperativeK64WideKernel<Rows>{},
        dim3(static_cast<unsigned>(tiles)), dim3(512), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows,int Depth>
struct DecodeNnCooperativeN256Kernel
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
            (block - group * tiles_per_group) * 256};
        decode_nn_cooperative_n256::run<Rows,Depth>(p);
    }
};

template<int Rows,int Depth>
CK_TILE_HOST int launch_decode_cooperative_n256(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_n256::Geometry<Rows,32,Depth>;
    const auto kernel = make_kernel<512, 1>(DecodeNnCooperativeN256Kernel<Rows,Depth>{},
        dim3(static_cast<unsigned>(tiles)), dim3(512), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows,int Depth>
struct DecodeNnBReuseN256Kernel
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
            (block - group * tiles_per_group) * 256};
        decode_nn_cooperative_n256::run<Rows,Depth,true>(p);
    }
};

template<int Rows,int Depth>
CK_TILE_HOST int launch_decode_cooperative_n256_b_reuse(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_n256::Geometry<Rows,32,Depth,true>;
    const auto kernel = make_kernel<512, 1>(DecodeNnBReuseN256Kernel<Rows,Depth>{},
        dim3(static_cast<unsigned>(tiles)), dim3(512), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
struct DecodeNnLowLdsN128Kernel
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
        decode_nn_cooperative_n256::run<Rows,2,true,128>(p);
    }
};

template<int Rows>
CK_TILE_HOST int launch_decode_cooperative_low_lds(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group)
{
    using G = decode_nn_cooperative_n256::Geometry<Rows,32,2,true,128>;
    const auto kernel = make_kernel<G::Waves*64, 1>(DecodeNnLowLdsN128Kernel<Rows>{},
        dim3(static_cast<unsigned>(tiles)), dim3(G::Waves*64), G::LdsBytes,
        cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}

template<int Rows>
CK_TILE_HOST int dispatch_decode_cooperative(void* args, int groups, hipStream_t stream,
                                             uint64_t tiles, int tiles_per_group, int variant)
{
    switch(variant)
    {
    case 0: return launch_decode_cooperative<Rows, 16, 4>(args, groups, stream, tiles, tiles_per_group);
    case 1: return launch_decode_cooperative<Rows, 16, 8>(args, groups, stream, tiles, tiles_per_group);
    case 2: return launch_decode_cooperative<Rows, 32, 4>(args, groups, stream, tiles, tiles_per_group);
    case 3: return launch_decode_cooperative<Rows, 32, 5>(args, groups, stream, tiles, tiles_per_group);
    case 4: return launch_decode_cooperative_compact<Rows>(args, groups, stream, tiles, tiles_per_group);
    case 5:
        if constexpr(Rows >= 2)
            return launch_decode_cooperative_wide<Rows>(args, groups, stream, tiles, tiles_per_group);
        return -1;
    case 6: return launch_decode_cooperative_compact_phase<Rows>(args, groups, stream, tiles, tiles_per_group);
    case 7:
        if constexpr(Rows >= 2)
            return launch_decode_cooperative_wide_phase<Rows>(args, groups, stream, tiles, tiles_per_group);
        return -1;
    case 8: return launch_decode_cooperative_k64<Rows>(args, groups, stream, tiles, tiles_per_group);
    case 9:
        if constexpr(Rows >= 2)
            return launch_decode_cooperative_k64_wide<Rows>(args, groups, stream, tiles, tiles_per_group);
        return -1;
    case 10: return launch_decode_cooperative_n256<Rows,2>(args, groups, stream, tiles, tiles_per_group);
    case 11: return launch_decode_cooperative_n256<Rows,3>(args, groups, stream, tiles, tiles_per_group);
    case 12: return launch_decode_cooperative_n256_b_reuse<Rows,2>(args, groups, stream, tiles, tiles_per_group);
    case 13: return launch_decode_cooperative_n256_b_reuse<Rows,3>(args, groups, stream, tiles, tiles_per_group);
    case 14: return launch_decode_cooperative_low_lds<Rows>(args, groups, stream, tiles, tiles_per_group);
    default: return -1;
    }
}
} // namespace ck_tile

extern "C" int ck_tile_hcu_grouped_gemm_bf16_nn_decode_cooperative_device_args(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group,
    int m, int n, int k, int variant)
{
    if(args == nullptr || groups < 1 || groups > 1024 || m < 1 || m > 64 ||
       variant < 0 || variant > 14 || ((variant == 5 || variant == 7) && m <= 16) || n < 1024 || n > 4096 || n % (variant >= 10 && variant != 14 ? 256 : 128) != 0 ||
       k < 2048 || k > 8192 || k % 128 != 0 || tiles_per_group != n / (variant >= 10 && variant != 14 ? 256 : 128) ||
       tiles != static_cast<uint64_t>(groups) * tiles_per_group) return -1;
    if(m <= 16) return ck_tile::dispatch_decode_cooperative<1>(args, groups, stream, tiles, tiles_per_group, variant);
    if(m <= 32) return ck_tile::dispatch_decode_cooperative<2>(args, groups, stream, tiles, tiles_per_group, variant);
    return ck_tile::dispatch_decode_cooperative<4>(args, groups, stream, tiles, tiles_per_group, variant);
}

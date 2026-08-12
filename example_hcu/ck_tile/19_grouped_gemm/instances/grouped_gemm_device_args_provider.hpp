// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "grouped_gemm_impl.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"

namespace grouped_gemm_device_args_detail {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename PrecType>
struct RegistryV3WarpRakedMOnlyTail
    : GemmConfigComputeV3RegPrefetch256W16Group8Aligned<PrecType>
{
    static constexpr bool kPadM = true;
};

template <typename PrecType>
struct RegistryV4M4M01_4MOnlyTail : GemmConfigComputeV4SquareM4M01_4<PrecType>
{
    static constexpr bool kPadM = true;
};

template <typename Config,
          typename PrecType,
          typename ALayout,
          typename BLayout,
          ck_tile::GroupedGemmMFilter MFilter = ck_tile::GroupedGemmMFilter::All>
int launch_one(void* device_args,
               int group_count,
               std::uint32_t num_cu,
               hipStream_t stream)
{
    grouped_gemm_tileloop<Config,
                          ALayout,
                          BLayout,
                          Row,
                          PrecType,
                           PrecType,
                           float,
                           PrecType,
                           0,
                           MFilter>(ck_tile::stream_config{stream},
                                   group_count,
                                   device_args,
                                   {},
                                   false,
                                   num_cu);
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
}

template <typename Config,
          typename PrecType,
          ck_tile::GroupedGemmMFilter MFilter = ck_tile::GroupedGemmMFilter::All>
int launch_layout(ck_tile::GroupedGemmLayout layout,
                  void* device_args,
                  int group_count,
                  std::uint32_t num_cu,
                  hipStream_t stream)
{
    if(layout == ck_tile::GroupedGemmLayout::nt)
        return launch_one<Config, PrecType, Row, Col, MFilter>(
            device_args, group_count, num_cu, stream);
    if(layout == ck_tile::GroupedGemmLayout::nn)
        return launch_one<Config, PrecType, Row, Row, MFilter>(
            device_args, group_count, num_cu, stream);
    if(layout == ck_tile::GroupedGemmLayout::tn)
        return launch_one<Config, PrecType, Col, Row, MFilter>(
            device_args, group_count, num_cu, stream);
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
}

template <typename AlignedConfig,
          typename TailConfig,
          typename PrecType,
          typename ALayout,
          typename BLayout>
int launch_m_only_hybrid_one(void* device_args,
                             int group_count,
                             std::uint32_t num_cu,
                             hipStream_t stream)
{
    int status = launch_one<AlignedConfig,
                             PrecType,
                             ALayout,
                             BLayout,
                             ck_tile::GroupedGemmMFilter::Aligned>(
        device_args, group_count, num_cu, stream);
    if(status != 0)
        return status;
    return launch_one<TailConfig,
                       PrecType,
                       ALayout,
                       BLayout,
                       ck_tile::GroupedGemmMFilter::Unaligned>(
        device_args, group_count, num_cu, stream);
}

template <typename AlignedConfig,
          typename TailConfig,
          typename PrecType>
int launch_m_only_hybrid(ck_tile::GroupedGemmLayout layout,
                         void* device_args,
                         int group_count,
                         std::uint32_t num_cu,
                         hipStream_t stream)
{
    if(layout == ck_tile::GroupedGemmLayout::nt)
        return launch_m_only_hybrid_one<
            AlignedConfig, TailConfig, PrecType, Row, Col>(
            device_args, group_count, num_cu, stream);
    if(layout == ck_tile::GroupedGemmLayout::nn)
        return launch_m_only_hybrid_one<
            AlignedConfig, TailConfig, PrecType, Row, Row>(
            device_args, group_count, num_cu, stream);
    if(layout == ck_tile::GroupedGemmLayout::tn)
        return launch_m_only_hybrid_one<
            AlignedConfig, TailConfig, PrecType, Col, Row>(
            device_args, group_count, num_cu, stream);
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
}

template <typename PrecType>
int launch_registry_candidate(ck_tile::GroupedGemmInstanceId instance_id,
                              ck_tile::GroupedGemmLayout layout,
                              void* device_args,
                              int group_count,
                              std::uint32_t num_cu,
                              hipStream_t stream)
{
    switch(instance_id)
    {
    case ck_tile::GroupedGemmInstanceId::none:
    case ck_tile::GroupedGemmInstanceId::bw_family_selected:
    case ck_tile::GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8:
    case ck_tile::GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3:
    case ck_tile::GroupedGemmInstanceId::bw_family_blas_transposed:
    case ck_tile::GroupedGemmInstanceId::bw_family_logical_k_tail: break;
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_GFX936)
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_256_m_only_padding:
        if(layout == ck_tile::GroupedGemmLayout::nt)
            return launch_m_only_hybrid_one<
                GemmConfigComputeV3RegPrefetch256W16Group8Aligned<PrecType>,
                RegistryV3WarpRakedMOnlyTail<PrecType>,
                PrecType,
                Row,
                Col>(device_args, group_count, num_cu, stream);
        break;
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding:
        if(layout == ck_tile::GroupedGemmLayout::nn)
            return launch_m_only_hybrid_one<
                GemmConfigComputeV3DsreadmStage256W8<PrecType>,
                RegistryV4M4M01_4MOnlyTail<PrecType>,
                PrecType,
                Row,
                Row>(device_args, group_count, num_cu, stream);
        if(layout == ck_tile::GroupedGemmLayout::tn)
            return launch_m_only_hybrid_one<
                GemmConfigComputeV3DsreadmStage256W8<PrecType>,
                RegistryV4M4M01_4MOnlyTail<PrecType>,
                PrecType,
                Col,
                Row>(device_args, group_count, num_cu, stream);
        break;
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding:
        if(layout == ck_tile::GroupedGemmLayout::nn)
            return launch_m_only_hybrid_one<
                GemmConfigComputeV3RegPrefetch256W16Group8DefaultPolicyAllLayouts<PrecType>,
                RegistryV4M4M01_4MOnlyTail<PrecType>,
                PrecType,
                Row,
                Row>(device_args, group_count, num_cu, stream);
        break;
#else
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_256_m_only_padding:
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding:
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding: break;
#endif
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_GFX938)
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_large_256:
        return launch_layout<GemmConfigMls256<PrecType>, PrecType>(
            layout, device_args, group_count, num_cu, stream);
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_small_128:
        if(layout == ck_tile::GroupedGemmLayout::nt)
            return launch_one<GemmConfigMls<PrecType>, PrecType, Row, Col>(
                device_args, group_count, num_cu, stream);
        break;
#else
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_large_256:
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_small_128: break;
#endif
    case ck_tile::GroupedGemmInstanceId::v4_128_m_only_padding:
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_GFX936)
        if(layout != ck_tile::GroupedGemmLayout::nn)
            return launch_m_only_hybrid<
                GemmConfigComputeV4SquareM4M01_4<PrecType>,
                RegistryV4M4M01_4MOnlyTail<PrecType>,
                PrecType>(layout, device_args, group_count, num_cu, stream);
#endif
        return launch_m_only_hybrid<GemmConfigComputeV4SquareM4<PrecType>,
                                    GemmConfigComputeV4SquareM4MOnlyPad<PrecType>,
                                    PrecType>(
            layout, device_args, group_count, num_cu, stream);
    case ck_tile::GroupedGemmInstanceId::v4_128_full_padding:
        return launch_layout<GemmConfigComputeV4SquareWideMpad<PrecType>,
                             PrecType>(layout, device_args, group_count, num_cu, stream);
    case ck_tile::GroupedGemmInstanceId::v4_64_nonpadding:
        return launch_layout<GemmConfigComputeV4<PrecType>, PrecType>(
            layout, device_args, group_count, num_cu, stream);
    case ck_tile::GroupedGemmInstanceId::v4_64_full_padding:
        return launch_layout<GemmConfigComputeV4Mpad<PrecType>, PrecType>(
            layout, device_args, group_count, num_cu, stream);
    }
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
}

} // namespace grouped_gemm_device_args_detail

// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "grouped_gemm_impl.hpp"

#if defined(CK_TILE_GROUPED_GEMM_TUNE_MLS)
template <typename PrecType>
using GroupedGemmFp16Config = GemmConfigMls<PrecType>;
template <typename PrecType>
using GroupedGemmFp16PaddedConfig = GemmConfigMls<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TUNE_SQUARE_M4)
template <typename PrecType>
using GroupedGemmFp16Config = GemmConfigComputeV4SquareM4<PrecType>;
template <typename PrecType>
using GroupedGemmFp16PaddedConfig = GemmConfigComputeV4SquareM4Mpad<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TUNE_SQUARE)
template <typename PrecType>
using GroupedGemmFp16Config = GemmConfigComputeV4Square<PrecType>;
template <typename PrecType>
using GroupedGemmFp16PaddedConfig = GemmConfigComputeV4SquareMpad<PrecType>;
#else
template <typename PrecType>
using GroupedGemmFp16Config = GemmConfigComputeV4<PrecType>;
template <typename PrecType>
using GroupedGemmFp16PaddedConfig = GemmConfigComputeV4Mpad<PrecType>;
#endif

int grouped_gemm_c_run_fp16(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream)
{
#if !defined(CK_TILE_GROUPED_GEMM_TUNE_MLS) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_SQUARE) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_SQUARE_M4) && \
    !defined(CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
    if(grouped_gemm_prefers_large_config<GemmConfigComputeV4Square<ck_tile::half_t>>(
           descs, group_count))
    {
        const auto hcu_target = ck_tile::get_hcu_target_enum();
        if(hcu_target == ck_tile::hcu_target_enum::gfx936 ||
           hcu_target == ck_tile::hcu_target_enum::gfx938)
        {
            return dispatch_grouped_gemm_c_layout<GemmConfigComputeV4SquareM4<ck_tile::half_t>,
                                                  ck_tile::half_t>(
                descs, group_count, a_layout, b_layout, workspace, stream);
        }
        return dispatch_grouped_gemm_c_layout<GemmConfigComputeV4Square<ck_tile::half_t>,
                                              ck_tile::half_t>(
            descs, group_count, a_layout, b_layout, workspace, stream);
    }
#endif
    return dispatch_grouped_gemm_c_layout<GroupedGemmFp16Config<ck_tile::half_t>, ck_tile::half_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_c_run_fp16_mpad(const ck_tile_dcu_grouped_gemm_desc* descs,
                                  int group_count,
                                  char a_layout,
                                  char b_layout,
                                  void* workspace,
                                  hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<GroupedGemmFp16PaddedConfig<ck_tile::half_t>,
                                          ck_tile::half_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_example_run_fp16(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    if(grouped_gemm_example_needs_padding<GroupedGemmFp16Config<ck_tile::half_t>>(argc, argv))
    {
        return run_gemm_example_prec_type<GroupedGemmFp16PaddedConfig<ck_tile::half_t>,
                                          ck_tile::half_t>(a_layout,
                                                           b_layout,
                                                           argc,
                                                           argv);
    }
#if !defined(CK_TILE_GROUPED_GEMM_TUNE_MLS) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_SQUARE) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_SQUARE_M4) && \
    !defined(CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
    if(grouped_gemm_example_prefers_large_config<GemmConfigComputeV4Square<ck_tile::half_t>>(
           argc, argv))
    {
        const auto hcu_target = ck_tile::get_hcu_target_enum();
        if(hcu_target == ck_tile::hcu_target_enum::gfx936 ||
           hcu_target == ck_tile::hcu_target_enum::gfx938)
        {
            return run_gemm_example_prec_type<GemmConfigComputeV4SquareM4<ck_tile::half_t>,
                                              ck_tile::half_t>(
                a_layout, b_layout, argc, argv);
        }
        return run_gemm_example_prec_type<GemmConfigComputeV4Square<ck_tile::half_t>,
                                          ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
#endif
    return run_gemm_example_prec_type<GroupedGemmFp16Config<ck_tile::half_t>, ck_tile::half_t>(
        a_layout, b_layout, argc, argv);
}

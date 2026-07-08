// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>
#include <utility>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_abd.hpp"
#include "ck/utility/common_header.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename LayoutTuple, typename Layout, index_t... Is>
static constexpr bool is_layout_tuple_uniform(std::integer_sequence<index_t, Is...>)
{
    return (true && ... && is_same_v<remove_cvref_t<tuple_element_t<Is, LayoutTuple>>, Layout>);
}

template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename CLayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          typename LDSTypeA                           = ComputeTypeA,
          typename LDSTypeB                           = ComputeTypeB>
struct DeviceGemmMultiABD_Xdl_CShuffle_V3
    : public DeviceGemmMultiABDV2R1<AsLayout,
                                    BsLayout,
                                    DsLayout,
                                    CLayout,
                                    AsDataType,
                                    BsDataType,
                                    DsDataType,
                                    CDataType,
                                    AElementwiseOperation,
                                    BElementwiseOperation,
                                    CElementwiseOperation>
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                      BlkGemmPipelineVer == BlockGemmPipelineVersion::v3,
                  "M2-2 HCU MultiABD correctness path currently enables v1/v3 pipelines");

    using ALayout = remove_cvref_t<tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<tuple_element_t<0, BsLayout>>;

    static_assert(is_layout_tuple_uniform<AsLayout, ALayout>(
                      std::make_integer_sequence<index_t, NumATensor>{}),
                  "HCU MultiABD V3 currently requires all A layouts to match A0");
    static_assert(is_layout_tuple_uniform<BsLayout, BLayout>(
                      std::make_integer_sequence<index_t, NumBTensor>{}),
                  "HCU MultiABD V3 currently requires all B layouts to match B0");

    static constexpr auto WarpTileConfig64 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               true>();
    static constexpr auto WarpTileConfig32 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               false>();
    static constexpr auto NXdlPerWave64    = WarpTileConfig64.At(Number<3>{});
    static constexpr auto NXdlPerWave32    = WarpTileConfig32.At(Number<3>{});

    template <typename WarpTileConfig>
    using GridwiseGemmBase = GridwiseGemmMultiABD_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        CLayout,
        AsDataType,
        BsDataType,
        GemmAccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        WarpTileConfig::At(Number<0>{}),
        WarpTileConfig::At(Number<1>{}),
        WarpTileConfig::At(Number<2>{}),
        WarpTileConfig::At(Number<3>{}),
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        WarpTileConfig::At(Number<4>{}),
        WarpTileConfig::At(Number<5>{}),
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        LDSTypeA,
        LDSTypeB>;

    using GridwiseGemm64 = GridwiseGemmBase<decltype(WarpTileConfig64)>;
    using GridwiseGemm32 = GridwiseGemmBase<decltype(WarpTileConfig32)>;
    using Argument       = typename GridwiseGemm64::Argument;

    struct Invoker : public BaseInvoker
    {
        template <typename GridwiseGemm>
        float RunImp(const typename GridwiseGemm::Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            const index_t k_grain = arg.KBatch * KPerBlock;
            const index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;
            const bool has_main_k_block_loop =
                GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            constexpr index_t minimum_occupancy = []() {
                if constexpr(BlkGemmPipeSched == BlockGemmPipelineScheduler::Interwave)
                {
                    return 2;
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    return (MPerBlock * NPerBlock / BlockSize <= 128) ? 2 : 1;
                }
                else
                {
                    return 1;
                }
            }();

            float ave_time = 0;
            const auto run_kernel = [&](const auto& kernel) {
                if(arg.KBatch > 1)
                {
                    hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
                                                     0,
                                                     arg.M * arg.N * sizeof(CDataType),
                                                     stream_config.stream_id_));
                }

                ave_time = launch_and_time_kernel(
                    stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
            };

            if(has_main_k_block_loop)
            {
                if(arg.KBatch > 1)
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        true,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        true,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
            }
            else
            {
                if(arg.KBatch > 1)
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        false,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        false,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
            }

            return ave_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            return RunImp<GridwiseGemm64>(arg, stream_config);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static auto MakeArgument(std::array<const void*, NumATensor> p_as,
                             std::array<const void*, NumBTensor> p_bs,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             std::array<index_t, NumATensor> StrideAs,
                             std::array<index_t, NumBTensor> StrideBs,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_as,
                        p_bs,
                        p_ds,
                        p_c,
                        M,
                        N,
                        K,
                        StrideAs,
                        StrideBs,
                        StrideDs,
                        StrideC,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(is_gfx11_supported() && arg.KBatch > 1)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: gfx11 does not support "
                             "KBatch > 1"
                          << std::endl;
            }
            return false;
        }

        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> && arg.KBatch > 1)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: bf16 atomic is not "
                             "supported for KBatch > 1"
                          << std::endl;
            }
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: K=" << arg.K
                          << " must be divisible by AK1=" << AK1 << " and BK1=" << BK1
                          << " without K padding" << std::endl;
            }
            return false;
        }

        if(CDEShuffleBlockTransferScalarPerVectors{}[Number<0>{}] <= 1 && arg.KBatch > 1)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: CDE shuffle vector for "
                             "C must be greater than 1 when KBatch > 1"
                          << std::endl;
            }
            return false;
        }

        if(get_warp_size() != 64)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: M2-2 HCU path expects "
                             "wave64"
                          << std::endl;
            }
            return false;
        }

        if constexpr(NXdlPerWave64 > 0)
        {
            return GridwiseGemm64::CheckValidity(arg);
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: no valid wave64 warp "
                             "tile"
                          << std::endl;
            }
            return false;
        }
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                                                      std::array<const void*, NumBTensor> p_bs,
                                                      std::array<const void*, NumDTensor> p_ds,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      std::array<index_t, NumATensor> StrideAs,
                                                      std::array<index_t, NumBTensor> StrideBs,
                                                      std::array<index_t, NumDTensor> StrideDs,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(MakeArgument(p_as,
                                                       p_bs,
                                                       p_ds,
                                                       p_c,
                                                       M,
                                                       N,
                                                       K,
                                                       StrideAs,
                                                       StrideBs,
                                                       StrideDs,
                                                       StrideC,
                                                       KBatch,
                                                       a_element_op,
                                                       b_element_op,
                                                       c_element_op));
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream{};
        str << "DeviceGemmMultiABD_Xdl_CShuffle_V3"
            << "<" << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0] << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0] << ">"
            << " A#: " << NumATensor << ", "
            << "B#: " << NumBTensor << ", "
            << "D#: " << NumDTensor << ", "
            << "BlkSize: " << BlockSize << ", "
            << "BlkTile: " << MPerBlock << "x" << NPerBlock << "x" << KPerBlock << ", "
            << "WaveTile: " << MPerXDL << "x" << NPerXDL << ", "
            << "WaveMap: " << MXdlPerWave << "x" << NXdlPerWave << ", "
            << "Pipeline: "
            << (BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ? "v1" : "v3");
        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

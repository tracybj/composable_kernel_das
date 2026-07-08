// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/utility/common_header.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// HCU V3 seed for the MultiABD API shape.
//
// This first adapter intentionally supports singleton A/B tuples only and delegates to the
// already validated HCU MultiD implementation. True NumATensor/NumBTensor > 1 support needs
// tuple A/B descriptors, tuple global buffers, and v7r2 A/B transfers in gridwise.
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

    static_assert(NumATensor == 1,
                  "DeviceGemmMultiABD_Xdl_CShuffle_V3 adapter currently supports one A tensor");
    static_assert(NumBTensor == 1,
                  "DeviceGemmMultiABD_Xdl_CShuffle_V3 adapter currently supports one B tensor");

    using ALayout = remove_cvref_t<tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<tuple_element_t<0, BsDataType>>;

    using DelegateDeviceGemm = DeviceGemmMultiD_Xdl_CShuffle_V3<
        ALayout,
        BLayout,
        DsLayout,
        CLayout,
        ADataType,
        BDataType,
        DsDataType,
        CDataType,
        GemmAccDataType,
        CShuffleDataType,
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
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        LDSTypeA,
        LDSTypeB>;

    using Argument = typename DelegateDeviceGemm::Argument;
    using Invoker  = typename DelegateDeviceGemm::Invoker;

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
        return DelegateDeviceGemm::MakeArgument(p_as[0],
                                                p_bs[0],
                                                p_ds,
                                                p_c,
                                                M,
                                                N,
                                                K,
                                                StrideAs[0],
                                                StrideBs[0],
                                                StrideDs,
                                                StrideC,
                                                KBatch,
                                                a_element_op,
                                                b_element_op,
                                                c_element_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return DelegateDeviceGemm::IsSupportedArgument(arg);
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
        str << "DeviceGemmMultiABD_Xdl_CShuffle_V3(singleton_ab_adapter) -> "
            << DelegateDeviceGemm{}.GetTypeString();
        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

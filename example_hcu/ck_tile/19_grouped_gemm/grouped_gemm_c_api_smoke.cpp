// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

template <typename DataType>
bool run_nn_variable_k_c_api_smoke(int dtype)
{
    const std::vector<int> ms{128, 256, 192};
    const std::vector<int> ns{128, 128, 256};
    const std::vector<int> ks{128, 192, 256};
    const int group_count = static_cast<int>(ms.size());

    std::vector<std::vector<DataType>> host_as(group_count);
    std::vector<std::vector<DataType>> host_bs(group_count);
    std::vector<std::vector<DataType>> host_cs(group_count);
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_as;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_bs;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_cs;
    std::vector<ck_tile_dcu_grouped_gemm_desc> descs;

    device_as.reserve(group_count);
    device_bs.reserve(group_count);
    device_cs.reserve(group_count);
    descs.reserve(group_count);

    const DataType one = ck_tile::type_convert<DataType>(1.0f);
    for(int i = 0; i < group_count; ++i)
    {
        host_as[i].assign(static_cast<std::size_t>(ms[i]) * ks[i], one);
        host_bs[i].assign(static_cast<std::size_t>(ks[i]) * ns[i], one);
        host_cs[i].resize(static_cast<std::size_t>(ms[i]) * ns[i]);

        device_as.push_back(
            std::make_unique<ck_tile::DeviceMem>(host_as[i].size() * sizeof(DataType)));
        device_bs.push_back(
            std::make_unique<ck_tile::DeviceMem>(host_bs[i].size() * sizeof(DataType)));
        device_cs.push_back(
            std::make_unique<ck_tile::DeviceMem>(host_cs[i].size() * sizeof(DataType)));

        device_as.back()->ToDevice(host_as[i].data());
        device_bs.back()->ToDevice(host_bs[i].data());
        device_cs.back()->SetZero();

        descs.push_back({device_as.back()->GetDeviceBuffer(),
                         device_bs.back()->GetDeviceBuffer(),
                         device_cs.back()->GetDeviceBuffer(),
                         1,
                         ms[i],
                         ns[i],
                         ks[i],
                         ks[i],
                         ns[i],
                         ns[i],
                         0,
                         nullptr,
                         nullptr});
    }

    ck_tile::DeviceMem workspace(
        ck_tile_dcu_grouped_gemm_workspace_size(group_count));
    const int rc = ck_tile_dcu_grouped_gemm_run(descs.data(),
                                                 group_count,
                                                 dtype,
                                                 'R',
                                                 'R',
                                                 workspace.GetDeviceBuffer(),
                                                 nullptr);
    if(rc != 0)
    {
        std::cerr << "ck_tile_dcu_grouped_gemm_run returned " << rc << std::endl;
        return false;
    }

    const hipError_t sync_status = hipDeviceSynchronize();
    if(sync_status != hipSuccess)
    {
        std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(sync_status) << std::endl;
        return false;
    }

    for(int i = 0; i < group_count; ++i)
    {
        device_cs[i]->FromDevice(host_cs[i].data());
        const float expected = static_cast<float>(ks[i]);
        for(std::size_t j = 0; j < host_cs[i].size(); ++j)
        {
            const float actual = ck_tile::type_convert<float>(host_cs[i][j]);
            if(std::abs(actual - expected) > 1.0f)
            {
                std::cerr << "group " << i << " output[" << j << "] expected " << expected
                          << ", got " << actual << std::endl;
                return false;
            }
        }
    }

    return true;
}

int main()
{
    const bool fp16_ok =
        run_nn_variable_k_c_api_smoke<ck_tile::half_t>(CK_TILE_DCU_GROUPED_GEMM_FP16);
    const bool bf16_ok =
        run_nn_variable_k_c_api_smoke<ck_tile::bf16_t>(CK_TILE_DCU_GROUPED_GEMM_BF16);

    if(fp16_ok && bf16_ok)
    {
        std::cout << "FP16/BF16 grouped GEMM C ABI NN variable-K smoke: PASS" << std::endl;
        return 0;
    }
    return 1;
}

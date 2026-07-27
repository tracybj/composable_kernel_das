// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

// Launch the selected gfx936 FP16/BF16 grouped-GEMM kernel from an existing
// device array of ck_tile::GemmTransKernelArg. This keeps descriptor creation
// in the caller while sharing the exact CK pipeline and epilogue instance.
extern "C" int ck_tile_hcu_grouped_gemm_gfx936_fp16_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream);

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_fp16_nn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream);

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_fp16_tn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream);

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_bf16_nn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream);

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_bf16_tn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream);

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_bf16_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream);

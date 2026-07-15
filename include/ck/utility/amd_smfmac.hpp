// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Hygon DCU: SMF (Sparse Matrix Fused) MFMA is NOT available.
// Upstream uses __builtin_amdgcn_smfmac_f32_16x16x32_f16/bf16 (gfx94x only).
// DCU uses MMAC (__builtin_hcu_mmac_*), not SMF.
// For sparse GEMM on DCU, use the MMAC-based warp gemm implementations.

#pragma once

namespace ck {

// intrin_smfmac_*: not supported on DCU.
// DCU has MMAC, not SMF. Use warp_gemm_attribute_mmac_impl.hpp instead.

} // namespace ck

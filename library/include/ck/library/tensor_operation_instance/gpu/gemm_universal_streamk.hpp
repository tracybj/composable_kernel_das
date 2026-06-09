// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.
//
// Adapted for Hygon DCU: gemm_universal_streamk stub.
// Upstream this depends on DeviceGemm_Streamk_V2 + device_gemm_xdl_cshuffle_streamk_v3.hpp
// with 250+ instance files. Not yet ported to DCU.
//
// DCU has ck_tile-level streamk infrastructure (streamk_common.hpp, tile partitioner)
// but not the classic-CK library instances.
//
// TODO: Port the V3 streamk pipeline:
//   - include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp
//   - library/src/tensor_operation_instance/gpu/gemm_universal_streamk/ (250+ files)

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// gemm_universal_streamk is not yet available on DCU.
// Requires V3 streamk pipeline import (~250 instance files).
// See gemm_universal.hpp for the non-streamk GEMM instances.

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

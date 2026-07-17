# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx938")
    add_example_executable(tile_example_gemm_quant_tensor gemm_quant_tensor.cpp)
    target_compile_definitions(tile_example_gemm_quant_tensor PRIVATE PRIMUS_TURBO_GFX938=1)
    add_example_executable(tile_example_gemm_quant_rowcol gemm_quant_rowcol.cpp)
    target_compile_definitions(tile_example_gemm_quant_rowcol PRIVATE PRIMUS_TURBO_GFX938=1)
else()
    message(DEBUG "Skipping HCU tensorwise quant GEMM example for current target")
endif()

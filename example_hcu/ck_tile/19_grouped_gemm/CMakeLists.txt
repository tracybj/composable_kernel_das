# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

set(TILE_EXAMPLE_GROUPED_GEMM "tile_example_grouped_gemm")
message("adding ${TILE_EXAMPLE_GROUPED_GEMM}")

# Keep the development executable deterministic. Shared implementation bodies
# and archived tuning probes must not be discovered as standalone TUs.
set(INSTANCE_SRCS
    instances/grouped_gemm_bf16.cpp
    instances/grouped_gemm_bf8.cpp
    instances/grouped_gemm_fp16.cpp
    instances/grouped_gemm_fp8.cpp
    instances/grouped_gemm_int4.cpp
    instances/grouped_gemm_int8.cpp
    # The generic FP16/BF16 dispatchers contain runtime-gated references to
    # these providers on every architecture, so both TUs are link providers
    # even when the compiled target is not gfx936.
    instances/grouped_gemm_gfx936_bf16.cpp
    instances/grouped_gemm_gfx936_fp16.cpp)


ck_add_arch_vgpr_compile_options()
add_compile_options("SHELL: -Rpass-analysis=kernel-resource-usage")

add_executable(${TILE_EXAMPLE_GROUPED_GEMM}
    grouped_gemm.cpp
    ${INSTANCE_SRCS}
)
target_include_directories(${TILE_EXAMPLE_GROUPED_GEMM} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
option(CK_TILE_GROUPED_GEMM_RESOURCE_REPORT "Print grouped_gemm kernel resource usage during compile" OFF)
if(CK_TILE_GROUPED_GEMM_RESOURCE_REPORT)
    target_compile_options(${TILE_EXAMPLE_GROUPED_GEMM} PRIVATE "-Rpass-analysis=kernel-resource-usage")
endif()

function(add_grouped_gemm_fast_target target_name fast_prec_define instance_src)
    add_executable(${target_name} EXCLUDE_FROM_ALL
        grouped_gemm.cpp
        instances/${instance_src}
    )
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    ck_add_arch_vgpr_compile_options()
    add_compile_options("SHELL: -Rpass-analysis=kernel-resource-usage")
    target_compile_definitions(${target_name}
        PRIVATE
            CK_TILE_GROUPED_GEMM_FAST_BUILD
            CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
            ${fast_prec_define}
            ${ARGN})
    if(CK_TILE_GROUPED_GEMM_RESOURCE_REPORT)
        target_compile_options(${target_name} PRIVATE "-Rpass-analysis=kernel-resource-usage")
    endif()
endfunction()

add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_fast CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_fast CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp)
set(HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES)
# Keep only the three promoted gfx936 instances in normal and production targets.
if(GPU_TARGETS MATCHES "gfx936" AND NOT GPU_TARGETS MATCHES "gfx938|gfx946")
    list(APPEND HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES
        instances/grouped_gemm_gfx936_bf16_nn_fixed_srd_lds8_boundary_lds0_vmem0.cpp
        instances/grouped_gemm_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier.cpp
        instances/grouped_gemm_gfx936_bf16_tn_blas_transposed_logical_k_tail.cpp)
    target_sources(${TILE_EXAMPLE_GROUPED_GEMM}
        PRIVATE ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES})
    target_compile_definitions(${TILE_EXAMPLE_GROUPED_GEMM}
        PRIVATE CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
    target_sources(tile_example_grouped_gemm_bf16_fast
        PRIVATE
            instances/grouped_gemm_gfx936_bf16.cpp
            ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES})
    target_compile_definitions(tile_example_grouped_gemm_bf16_fast
        PRIVATE CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
endif()
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_baseline CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_baseline CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
# BW-family large aligned RC: retain only the selected V3
# 256x256x64/W16/GroupNum=8 reproduction targets. Groups 1/2/3/5/8 are
# correct; groups 1/3/4/5/8 performance scales with the same selected config.
# Removed probes: W8 K64 spilled and took 21.88-24.97 ms; 128x256 K64/K32
# reached 6.80/5.92 ms; W16 K32, M8xN2 and explicit overlap reached
# 6.52/5.87/5.42 ms; direct-store was neutral (5.1952 vs 5.1949 ms CShuffle).
# The selected warp-raked V3 policy reduced LDS bank conflicts by 68.09% and
# FP16 latency from 5.1980 to 5.1634 ms without changing V3/V4 implementations.
# The normal targets exercise the bounded default while V4 remains the fallback
# for other shapes, layouts and targets. Group-count tests cover 1/2/3/5/8.
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_v3_regprefetch_256_w16_group8 CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_v3_regprefetch_256_w16_group8 CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8)
# Historical tuning targets are intentionally absent from the production
# build graph. The normal targets above exercise the selected gfx936 defaults;
# the two explicit targets below reproduce the accepted gfx938 MLS policy.
# gfx938 tuning summary for group_count=4, M/N/K=4096/7168/4096:
# keep only the accepted 256x256x32, eight-wave, K32 single-stage MLS target.
# It overlaps the next global load with MMAC, skips the redundant post-MMAC
# sync, retains CShuffle, uses 32 KiB LDS and permits two blocks/CU. Standalone
# FP16/BF16 measured 4.77439/4.77946 ms and passed CPU reference validation.
# Removed candidates covered Compute-V3 register prefetch, K64/split-register,
# sequential/fused pairs, direct store, alternative CShuffle/warp/tile shapes,
# priority/sync/DS-addressing variants, Lit/Lts and non-persistent traversal.
# They were slower, neutral, unsupported, superseded, or failed Primus
# fwd/grad-A/grad-B correctness. The normal fp16/bf16 targets also exercise
# this accepted configuration through the bounded default dispatch.
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_mls_256_k32_single_stage CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_TUNE_MLS_256 CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD CK_TILE_GROUPED_GEMM_MLS_SKIP_POST_MMAC_SYNC)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_mls_256_k32_single_stage CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_TUNE_MLS_256 CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD CK_TILE_GROUPED_GEMM_MLS_SKIP_POST_MMAC_SYNC)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp8_fast  CK_TILE_GROUPED_GEMM_FAST_FP8  grouped_gemm_fp8.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf8_fast  CK_TILE_GROUPED_GEMM_FAST_BF8  grouped_gemm_bf8.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_int8_fast CK_TILE_GROUPED_GEMM_FAST_INT8 grouped_gemm_int8.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_int4_fast CK_TILE_GROUPED_GEMM_FAST_INT4 grouped_gemm_int4.cpp)

if(GPU_TARGETS MATCHES "gfx938|gfx946")
    set(HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS)
    if(GPU_TARGETS MATCHES "gfx938")
        list(APPEND HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS
             CK_TILE_HCU_QUANT_GEMM_GFX938=1)
    endif()
    if(GPU_TARGETS MATCHES "gfx946")
        list(APPEND HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS
             CK_TILE_HCU_QUANT_GEMM_GFX946=1)
    endif()

    add_executable(tile_example_grouped_gemm_quant_tensor EXCLUDE_FROM_ALL
        quant_grouped_gemm_tensor.cpp)
    target_compile_definitions(tile_example_grouped_gemm_quant_tensor
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})

    add_executable(tile_example_grouped_gemm_quant_rowcol EXCLUDE_FROM_ALL
        quant_grouped_gemm_rowcol.cpp)
    target_compile_definitions(tile_example_grouped_gemm_quant_rowcol
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})

    add_executable(tile_example_grouped_gemm_quant_blockwise EXCLUDE_FROM_ALL
        quant_grouped_gemm_blockwise.cpp)
    target_compile_definitions(tile_example_grouped_gemm_quant_blockwise
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})

    add_custom_target(tile_example_grouped_gemm_quant_smoke
        COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_tensor>
        COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_rowcol>
        COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_blockwise>
        DEPENDS
            tile_example_grouped_gemm_quant_tensor
            tile_example_grouped_gemm_quant_rowcol
            tile_example_grouped_gemm_quant_blockwise
        COMMENT "Running HCU ck_tile FP8/BF8 quant grouped GEMM smoke tests"
        VERBATIM)

    foreach(quant_mode tensor rowcol blockwise)
        add_test(NAME tile_example_grouped_gemm_quant_${quant_mode}_smoke
            COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_${quant_mode}>)
        set_tests_properties(tile_example_grouped_gemm_quant_${quant_mode}_smoke
            PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;quant;smoke")
    endforeach()
endif()

add_library(ck_tile_hcu_grouped_gemm STATIC EXCLUDE_FROM_ALL
    grouped_gemm.cpp
    instances/grouped_gemm_fp16.cpp
    instances/grouped_gemm_bf16.cpp
    instances/grouped_gemm_gfx936_fp16.cpp
    instances/grouped_gemm_gfx936_bf16.cpp
    ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
)
set_source_files_properties(
    ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
    PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-template;-Wno-unneeded-internal-declaration")
target_include_directories(ck_tile_hcu_grouped_gemm PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ck_tile_hcu_grouped_gemm
    PRIVATE
        CK_TILE_EXAMPLE_NO_MAIN
        CK_TILE_GROUPED_GEMM_FAST_BUILD
        CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
        CK_TILE_GROUPED_GEMM_FAST_FP16
        CK_TILE_GROUPED_GEMM_FAST_BF16)
if(HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES)
    # Keep gfx936-only declarations and link providers paired. In particular,
    # a gfx938-only build has an empty selected source list and must not make
    # grouped_gemm.cpp reference the gfx936 selected entry points.
    target_compile_definitions(ck_tile_hcu_grouped_gemm
        PRIVATE CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
endif()
set_target_properties(ck_tile_hcu_grouped_gemm PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Minimal consumer-independent library used by the public C ABI matrix. It
# deliberately excludes all default-off Phase316/317 candidate TUs so generic
# NT/NN/TN fallback coverage cannot accidentally resolve an experiment symbol.
add_library(ck_tile_hcu_grouped_gemm_generic STATIC EXCLUDE_FROM_ALL
    grouped_gemm.cpp
    instances/grouped_gemm_fp16.cpp
    instances/grouped_gemm_bf16.cpp
    instances/grouped_gemm_gfx936_fp16.cpp
    instances/grouped_gemm_gfx936_bf16.cpp
)
target_include_directories(ck_tile_hcu_grouped_gemm_generic PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ck_tile_hcu_grouped_gemm_generic
    PRIVATE
        CK_TILE_EXAMPLE_NO_MAIN
        CK_TILE_GROUPED_GEMM_FAST_BUILD
        CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
        CK_TILE_GROUPED_GEMM_FAST_FP16
        CK_TILE_GROUPED_GEMM_FAST_BF16)
set_target_properties(ck_tile_hcu_grouped_gemm_generic
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tile_example_grouped_gemm_c_api_smoke EXCLUDE_FROM_ALL
    grouped_gemm_c_api_smoke.cpp
)
target_link_libraries(tile_example_grouped_gemm_c_api_smoke
    PRIVATE ck_tile_hcu_grouped_gemm_generic)

# Production-selector variant of the exact same 60-case public C ABI matrix.
# On gfx936 it links the promoted selected instances; on gfx938 the selected
# source list is empty and the architecture-specific generic portfolio stays
# unchanged.
add_executable(tile_example_grouped_gemm_c_api_selected_smoke EXCLUDE_FROM_ALL
    grouped_gemm_c_api_smoke.cpp
)
target_link_libraries(tile_example_grouped_gemm_c_api_selected_smoke
    PRIVATE ck_tile_hcu_grouped_gemm)

add_executable(tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke EXCLUDE_FROM_ALL
    grouped_gemm_c_api_tn_logical_k_tail_smoke.cpp
)
target_link_libraries(tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke
    PRIVATE ck_tile_hcu_grouped_gemm)

# Host-only Phase-1 selector/contract test. This target deliberately does not
# link a grouped-GEMM provider, so adding it cannot alter a device launch.
add_executable(tile_example_grouped_gemm_selector_test EXCLUDE_FROM_ALL
    grouped_gemm_selector_test.cpp
)
add_test(NAME tile_example_grouped_gemm_selector_test
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_selector_test>)
set_tests_properties(tile_example_grouped_gemm_selector_test
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;selector")

set(GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS
    -group_count=2
    -Ms=128,256
    -Ns=128,128
    -Ks=128,128
    -stride_As=128,128
    -stride_Bs=128,128
    -stride_Cs=128,128
    -validate=1
    -warmup=1
    -repeat=1
)

set(GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS
    -group_count=3
    -Ms=128,256,192
    -Ns=128,128,256
    -Ks=128,192,256
    -stride_As=128,192,256
    -stride_Bs=128,192,256
    -stride_Cs=128,128,256
    -validate=1
    -warmup=1
    -repeat=1
)

set(GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS}
)
list(REMOVE_ITEM GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS -stride_Bs=128,192,256)
list(APPEND GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS -stride_Bs=128,128,256)

# TN 布局的对齐 variable-K 测试参数：A 为 ColumnMajor，B/C 为 RowMajor。
# ColumnMajor A 的 leading stride 是 M，所以 stride_As 与各组 Ms 相同。
set(GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS
    -group_count=3
    -Ms=128,256,192
    -Ns=128,128,256
    -Ks=128,192,256
    -stride_As=128,256,192
    -stride_Bs=128,128,256
    -stride_Cs=128,128,256
    -validate=1
    -warmup=1
    -repeat=1
)

# 全维 padding 测试故意选用 M/N/K 都不满足 tile 对齐的 shape，用于验证
# kPadM=true、kPadN=true、kPadK=true 时的边界 load/store 正确性。
set(GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS
    -group_count=3
    -Ms=65,129,191
    -Ns=129,130,257
    -Ks=129,193,255
    -stride_Cs=129,130,257
    -validate=1
    -warmup=1
    -repeat=1
)

# 按 NT、NN、TN 的物理布局分别设置 A/B 的 leading stride；调用方仍只分配
# 逻辑尺寸的 A/B/C buffer，不需要为 padded shape 额外申请显存。
set(GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS}
    -stride_As=129,193,255
    -stride_Bs=129,193,255
)
set(GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS}
    -stride_As=129,193,255
    -stride_Bs=129,130,257
)
set(GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS}
    -stride_As=65,129,191
    -stride_Bs=129,130,257
)

# 聚合 smoke target：构建后顺序运行 FP16/BF16 的 NT/NN/TN、variable-K、
# 全维 padding 以及 C ABI 正确性用例。它属于 example 测试，不是 ck4inductor 测试。
add_custom_target(tile_example_grouped_gemm_fp16_bf16_fast_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_c_api_smoke>
    DEPENDS
            tile_example_grouped_gemm_fp16_fast
            tile_example_grouped_gemm_bf16_fast
            tile_example_grouped_gemm_c_api_smoke
    COMMENT "Running HCU ck_tile grouped GEMM FP16/BF16 fast smoke tests"
    VERBATIM
)

add_test(NAME tile_example_grouped_gemm_fp16_fast_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
)
set_tests_properties(tile_example_grouped_gemm_fp16_fast_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;smoke"
)

add_test(NAME tile_example_grouped_gemm_bf16_fast_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
)
set_tests_properties(tile_example_grouped_gemm_bf16_fast_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;smoke"
)

# 将一个 example 命令注册成独立 CTest，便于通过 ctest -R 精确执行。
# a_layout/b_layout 使用 R/C 表示 RowMajor/ColumnMajor。
function(add_grouped_gemm_fast_smoke_test test_name target_name precision a_layout b_layout shape_args)
    add_test(NAME ${test_name}
        COMMAND $<TARGET_FILE:${target_name}>
                -prec=${precision}
                -a_layout=${a_layout}
                -b_layout=${b_layout}
                ${${shape_args}}
    )
    set_tests_properties(${test_name}
        PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;smoke"
    )
endfunction()

add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R R GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R R GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nt_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R C GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nt_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R C GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R R GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R R GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS)

# TN variable-K：验证新增加的 ColumnMajor-A + RowMajor-B layout。
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_tn_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 C R GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_tn_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 C R GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS)

# NT/NN/TN 全维 padding：分别覆盖 FP16 和 BF16 的非对齐 M/N/K。
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nt_padded_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R C GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nt_padded_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R C GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_padded_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R R GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_padded_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R R GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_tn_padded_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 C R GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_tn_padded_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 C R GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS)

# C ABI smoke 绕过 example CLI，直接验证公开 grouped GEMM C 接口的 layout/padding。
add_test(NAME tile_example_grouped_gemm_c_api_layout_padding_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_c_api_smoke>
)
set_tests_properties(tile_example_grouped_gemm_c_api_layout_padding_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;c_api;smoke"
)

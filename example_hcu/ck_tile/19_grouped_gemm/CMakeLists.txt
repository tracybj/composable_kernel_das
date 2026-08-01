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

# These wrappers instantiate Hygon gfx936-specific policies and must not be
# compiled into the gfx938 development executable.
if(GPU_TARGETS MATCHES "gfx936" AND NOT GPU_TARGETS MATCHES "gfx938|gfx946")
    list(APPEND INSTANCE_SRCS
        instances/grouped_gemm_gfx936_bf16_tn_direct.cpp)
endif()

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
# Keep only the two promoted gfx936 instances in normal and production targets.
if(GPU_TARGETS MATCHES "gfx936" AND NOT GPU_TARGETS MATCHES "gfx938|gfx946")
    list(APPEND HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES
        instances/grouped_gemm_gfx936_bf16_nn_fixed_srd_lds8_boundary_lds0_vmem0.cpp
        instances/grouped_gemm_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier.cpp)
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
# Phase-37 isolated W8 candidate: two K16 LDS slots (32 KiB), 8 waves, 212 VGPR,
# zero spill. FP16/BF16 are correct, but ~7.6 ms loses to W16 ~5.2 ms, so normal
# targets do not select it. Rejected probes: K32 spilled; 64-KiB K16 and M8N1
# were slower/resource-heavier. M2N4 is retained as the explicit reproducer.
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_v3_w8_overlap CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_W8_OVERLAP)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_v3_w8_overlap CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_W8_OVERLAP)
# BW-family 256x256x64/W8 exact operand schedule. RC retains the prior forward
# winner. NN/TN load non-K-contiguous inputs with the validated thread-raked
# transpose and write eight K4 vectors directly into the same swizzled LDS
# operand layout. At groups=4, NN is 4.932/4.925 ms and TN is
# 6.672/6.656 ms for FP16/BF16. All four dtype/layout cases pass reference
# validation; codegen is at most 255 VGPR, 64 KiB LDS and zero spill.
foreach(selected_dtype fp16 bf16)
    string(TOUPPER ${selected_dtype} selected_dtype_upper)
    set(bw_family_selected_target
        tile_example_grouped_gemm_${selected_dtype}_v3_bw_family_selected)
    add_grouped_gemm_fast_target(
        ${bw_family_selected_target}
        CK_TILE_GROUPED_GEMM_FAST_${selected_dtype_upper}
        grouped_gemm_${selected_dtype}.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_SPLIT_GLOBAL
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
        CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
    # Source-compatible build alias. The canonical executable is emitted with
    # the BW-family name; old automation can still request the gfx936 target.
    add_custom_target(
        tile_example_grouped_gemm_${selected_dtype}_v3_gfx936_selected
        DEPENDS ${bw_family_selected_target})
endforeach()
# BF16 NN scheduling probe: identical selected kernel and exact tile traversal,
# but launch the complete block grid instead of the 80-block persistent loop.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_selected_nonpersistent
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_SPLIT_GLOBAL
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_TRITON_NONPERSISTENT)
# hipBLASLt BF16 NN ISA reconstruction: four 16-K circular-layout stages,
# direct dwordx4 A/B producers, A-side ds_read_m32x16 and B-side ds_read_b64.
# The first gate refills one complete K64 tile at a time; circular overwrite
# overlap is enabled only after RR correctness and resource checks pass.
foreach(rr_blas_dtype fp16 bf16)
    string(TOUPPER ${rr_blas_dtype} rr_blas_dtype_upper)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_${rr_blas_dtype}_v3_gfx936_rr_blas_stage
        CK_TILE_GROUPED_GEMM_FAST_${rr_blas_dtype_upper}
        grouped_gemm_${rr_blas_dtype}.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
endforeach()
# Stage-circular BF16 follow-up. Each dead K16 slot is refilled from the next
# K64 tile after its LDS operands are resident in VGPRs. The intervening
# direct-to-LDS M0 sequence produces the BLAS-native accumulator order, so
# this target deliberately bypasses the full-refill block-tile remap.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
# hipBLASLt WGM8 steady-state schedule probe. It keeps the validated RR
# producer/consumer/output mapping, but leaves the next stage's eight LDS
# reads in flight across the current stage MMAC and uses the BLAS
# lgkmcnt(8)/vmcnt(4)/barrier cadence. The K64 boundary stays fully fenced.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_exact
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_EXACT_SCHEDULE)
# Continuous-K16 follow-up. Stage 3 reads the already-refilled stage 0 of the
# next K64 tile into the dead operand buffer, removing the per-K64 vmcnt(0),
# block barrier and stage-0 reload retained by the first exact probe.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS)
# hipBLASLt raises scheduler priority only across each 32-MMAC consumer
# interval. Test that instruction-level scheduling hint independently from
# the validated continuous-K16 candidate.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_setprio
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_SETPRIO)
# Keep each 32-MMAC interval contiguous before issuing the following stage's
# eight LDS reads, matching the ordering visible in the BLAS ELF.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT)
# Combine the two independently gated BLAS ISA hints after strict ordering
# showed a small positive result and setprio alone was neutral.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_setprio
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_SETPRIO)
# Launch every 256x256 output tile as an independent block while retaining
# the validated continuous K16 producer/consumer and transpose-C contract.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_nonpersistent
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C_NONPERSISTENT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS)
# Source-locality sweep. Keep the continuous K16 body identical and vary only
# the persistent M-tile grouping so its effect on TA/TCP pressure is isolated.
foreach(rr_m01 1 2 4 16)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_m01_${rr_m01}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=${rr_m01}
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS)
endforeach()
# hipBLASLt's selected solution reports WGM8. Sweep CK's coarse spatial
# workgroup grouping while holding M01=8 and the continuous K16 body fixed.
foreach(rr_group_num 2 4 8)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_group_${rr_group_num}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_GROUP_NUM=${rr_group_num}
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS)
endforeach()
# Combine the two independently positive Phase-59/61 changes.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT)
# hipBLASLt-style output probe. Compose the validated producer/LDS/consumer
# permutation into a direct per-accumulator coordinate and emit scalar
# buffer_store_short instructions, avoiding all four epilogue LDS rounds.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_store
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT)
# Keep the selected LDS transpose and coalesced output mapping, but split each
# 16-byte dwordx4 write into two 8-byte dwordx2 requests. This isolates request
# width from the scalar-store candidate's much larger instruction footprint.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_store_vec4
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_STORE_VEC4
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT)
# Force the source-level wait/barrier/MMAC order to survive LLVM scheduling.
# The vendor ISA synchronizes all eight waves before each 32-MMAC interval;
# the Phase-62 ISA had sunk that barrier below the MMACs.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_barrier_before_mmac
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC)
# Reproduce the vendor consumer boundary exactly: all waves rendezvous after
# the partial waits, the first MMAC issues at normal priority, and the
# remaining 31 MMACs run at priority 1.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_barrier_setprio_after_first
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST)
# Runtime grid-queue sweep on the Phase-62 body. With the environment variable
# unset this is bit-for-bit the same 80-block launch; values 2..16 only change
# the number of queued persistent blocks and their grid-stride tile assignment.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_grid_sweep
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME)
# Split the conservative synchronization into two independent probes so the
# minimum correctness requirement can be retained without destroying overlap.
foreach(wait_kind LDS VMEM)
    string(TOLOWER "${wait_kind}" wait_kind_lower)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_full_${wait_kind_lower}_wait_grid_sweep
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
        CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FULL_${wait_kind}_WAIT)
endforeach()
# Sweep intermediate lgkmcnt thresholds between the fast-but-racy count 8 and
# the safe full wait. Each target retains runtime grid scaling for the
# 240-block correctness stress gate.
foreach(wait_count 6 4 2)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_lds_wait_${wait_count}_grid_sweep
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
        CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_WAIT_COUNT=${wait_count})
endforeach()
# Pin the 32 MMACs in the vendor B-major operand-reuse order with inline ISA;
# the earlier C++ loop was still freely regrouped by LLVM's machine scheduler.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_inline_mmac_order_grid_sweep
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_INLINE_MMAC_ORDER)
# Combine all three vendor steady-state boundaries: barrier before the
# consumer interval, B-major inline MMAC issue order, and priority raise after
# the first MMAC. This is intentionally a separate candidate because each
# isolated probe was neutral/slower but their scheduler effects are coupled.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_vendor_consumer_grid_sweep
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_INLINE_MMAC_ORDER
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST)
# Reserve the vendor kernel's contiguous v0:v127 accumulator bank and issue
# the whole 32-MMAC stage from one asm region. Count 6 retains the repeated
# 240-block correctness guarantee established by the LDS-wait sweep.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_fixed_accum_grid_sweep
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_WAIT_COUNT=6
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS)
# Remap logical C tuples onto the allocator's physical tuple assignment so
# MMAC destinations become v0,v4,...,v124 in the same n-major reuse order as
# the custom hipBLASLt kernel. Keep this isolated from the older fixed-accum
# probe because it also applies the inverse mapping before the epilogue.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_linear_accum_grid_sweep
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_WAIT_COUNT=6
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS)
# Launch all 256 output tiles independently while retaining the Phase-62
# strict K16 body and M01=2 traversal. This isolates grid residency/tail
# utilization from the current 80-block persistent scheduler.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_nonpersistent
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C_NONPERSISTENT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SAFE_SYNC)
# LDS-address-generation ablation on the Phase-62 champion. Keep the global
# mapping and K16 schedule identical while scalarizing each wave's two m0
# bases once outside the dynamic K loop.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0)
# LDS-consumer address ablation on the scalar-m0 winner. Keep the four
# lane/wave/wrap-adjusted A/B bases live across the K loop, matching the custom
# BLAS ISA, and add only the compile-time K16 stage displacement at each read.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_fixed_lds
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_LDS_BASES)
# True scalar-SRD producer. Unlike the int32x4_t streaming probe, reserve the
# two mutable resources in s44:s51 so the K16 loop needs no v_readfirstlane
# reconstruction and matches the custom BLAS load/update instruction contract.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_fixed_stream_srd
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
# Pin all 32 fp32x4 accumulator tuples to the vendor kernel's physical
# v0:v127 bank. Keep the validated scalar-M0 producer and otherwise preserve
# the selected strict continuous schedule for an isolated A/B comparison.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_fixed_accum_v2
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2)
# Extend V2 by pinning the two alternating 24-VGPR LDS operand banks to the
# vendor layout: A0/A1 in v128:v159 and B0/B1 in v160:v175.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_fixed_accum_operands_v3
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_OPERANDS_V3)
# Joint fixed-register gate before moving the complete steady stage into one
# assembly scheduling region. This combines the already-correct scalar SRDs,
# accumulator bank and alternating operand banks without changing math/order.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_fixed_all_v5
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_OPERANDS_V3)
# First full-region step: V5 plus a single wait/barrier/MMAC consumer asm.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_consumer_asm_v6
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_OPERANDS_V3
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONSUMER_ASM_V6)
# V7 fixes each complete eight-read LDS packet directly into the alternate
# operand bank. Together with the fixed direct-refill asm and V6 consumer asm,
# the steady stage is now three scheduler-opaque packets with no scalarized
# fragment assignments between them.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_lds_consumer_asm_v7
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_OPERANDS_V3
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONSUMER_ASM_V6
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_ASM_V7)
# V4 keeps V3's physical register banks but puts the eight-fragment operand
# on MMAC source port 0 and the four-fragment operand on source port 1, exactly
# matching the custom BLAS ISA. Correctness is gated before any performance use.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_fixed_accum_operands_vendor_order_v4
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_OPERANDS_V3
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VENDOR_OPERAND_ORDER_V4)
# VMEM-depth sweep on the scalar-m0 winner. The validated default is vmcnt(4);
# larger counts retain more direct-to-LDS refills in flight and test whether
# the SQTT-attributed buffer-load issue backpressure can be reduced safely.
foreach(grouped_gemm_vmem_wait_count IN ITEMS 6 8)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_vmcnt${grouped_gemm_vmem_wait_count}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VMEM_WAIT_COUNT=${grouped_gemm_vmem_wait_count})
endforeach()
# Exact custom-BLAS WGM8 tile traversal on the scalar-m0 compute body:
# GroupNum=8 performs the XCD-interleaved remap and M01=8 groups eight M
# tiles before advancing N. Earlier M01-only sweeps retained GroupNum=1 and
# therefore did not reproduce this solution/index contract.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_g8_m01_8_scalar_m0
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_GROUP_NUM=8
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0)
# Split the exact WGM8 experiment into its two independent pieces. These
# targets retain the eight-XCD remap while sweeping only the local M01 order,
# so the result can distinguish cross-XCD distribution from WGM8 traversal.
foreach(grouped_gemm_wgm8_m01 IN ITEMS 1 2 4)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_g8_m01_${grouped_gemm_wgm8_m01}_scalar_m0
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_GROUP_NUM=8
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=${grouped_gemm_wgm8_m01}
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0)
endforeach()
# WGM8 is a full-grid launch contract in the custom BLAS solution. Re-test the
# pure eight-way mapping without an additional M01 swizzle and without the
# 80-block persistent stride, which otherwise changes the WGM8 block sequence.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_g8_m01_1_scalar_m0_nonpersistent
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_GROUP_NUM=8
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=1
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C_NONPERSISTENT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_TRITON_NONPERSISTENT)
# BLAS-style global address split on top of the validated scalar-m0 winner:
# keep the buffer resource and lane voffset stable while each K16 displacement
# uses the instruction's SGPR soffset field.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_soffset
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_SOFFSET)
# B-source descriptor/offset ablation on the scalar-m0 winner. Use the same
# structured-buffer idxen+offen contract visible in the custom WGM8 ISA while
# keeping normal dispatch and the validated control target unchanged.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_b_idxen
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_B_IDXEN)
# Exact solution-ELF SRD cadence on top of scalar-m0: both source resources
# advance after every direct K16 load pair, so the lane voffsets remain
# invariant and the scalar base/range updates occupy the load-to-wait gap.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_stream_resource
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE)
# Correctness/debug split for the streaming-SRD path. Each target advances
# only one operand resource so initial structured-address and raw-address
# semantics can be validated independently before recombining them.
foreach(grouped_gemm_stream_side IN ITEMS A B)
    string(TOLOWER "${grouped_gemm_stream_side}" grouped_gemm_stream_side_lower)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_stream_${grouped_gemm_stream_side_lower}_resource
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
        CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
        CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_${grouped_gemm_stream_side}_RESOURCE)
endforeach()
# Runtime-only queued-grid sweep on the validated scalar-m0 body. With the
# environment variable unset this is the same occupancy-derived 80-block grid.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_grid_sweep
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME)
# BLAS steady-state overlap: leave the direct-to-LDS refill in flight across
# the current 32 MMACs, then wait/synchronize before the next stage.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_postsync
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_POSTCOMPUTE_SYNC)
# Match hipBLASLt's exact priority boundary: first MMAC at normal priority,
# remaining 31 MMACs at priority 1.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_setprio_after_first
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST)
# Global-address-generation ablation on the Phase-62 champion. K16-uniform
# offsets advance the scalar resource while each lane keeps a stable voffset.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_resource
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_RESOURCE)
# Combine the validated scalar-m0 hot-loop contraction with the independent
# scalar-resource address split. Keep it separately selectable because the
# resource-only ablation was slightly slower on its own.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_scalar_m0_resource
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_RESOURCE)
# Match the BLAS consumer operand-reuse order (N outer, M inner) on top of the
# independently validated strict + M01=2 winner.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_gfx936_rr_blas_ring_continuous_strict_m01_2_blas_mmac
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
    CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
    CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=2
    CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
    CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_MMAC_ORDER)
# BLAS-ISA-aligned producer probe: replace the packed RC K64
# global->VGPR->LDS path with four K16-equivalent buffer_load_dwordx4 ... lds
# issues per operand. It is isolated from the selected/default binaries until
# correctness, resource usage and same-device latency all pass.
foreach(direct_dtype fp16 bf16)
    string(TOUPPER ${direct_dtype} direct_dtype_upper)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_${direct_dtype}_v3_gfx936_direct_g2l_full
        CK_TILE_GROUPED_GEMM_FAST_${direct_dtype_upper}
        grouped_gemm_${direct_dtype}.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
        CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01=8
        CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
endforeach()
# Retained DSReadM reproducers for comparing the older operand path.
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_v3_dsreadm CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_v3_dsreadm CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM)
# Staged DSReadM remains the FP16 TN winner at 5.554 ms. Its BF16 W8/W16
# variants measured about 17.0/16.2 ms combined and were not selected; the
# rejected W16 target/config was removed.
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_v3_dsreadm_stage CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_v3_dsreadm_stage CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE)
# hipBLASLt BF16 TN ISA reconstruction: retain the correctness-proven CK
# two-sided DSReadM consumer, but replace its K64 global->VGPR->LDS producer
# with four direct dwordx4->LDS stages. This target is intentionally isolated
# from default dispatch while correctness, resources and overlap are evaluated.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN)
# Four-slot K16 shadow pipeline matching the BLAS vmcnt(4)+barrier cadence.
# It remains independently switchable from the correctness-proven full-K64
# direct producer above.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR)
# Scalarized LDS base plus rolling global offsets. This is an ISA-level
# ablation of the circular producer and remains disabled in normal dispatch.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_m0
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_M0_ROLLING)
# Scalar buffer-resource stepping with stable vector offsets, mirroring the
# BLAS hot-loop address split. Kept separate from the rolling-voffset probe.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE)
# Full-grid scheduler ablation on the best scalar-resource compute body.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_nonpersistent
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_STAGE_NONPERSISTENT)
# Double-buffer K16 operand fragments so DSReadM for the next stage can overlap
# the current MMAC group. This remains an isolated persistent-scheduler probe.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_operand_pipeline
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED)
# B-only operand look-ahead: overlap two next-stage LDS reads without extending
# the lifetime of all six A+B fragments.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED)
# Exact hipBLASLt grad-W shared-memory contract: four interleaved 16-KiB
# [A|B] stages, the paired-wave m0 swizzle, and the ELF-derived DSReadM
# address map. Keep it isolated until correctness and PMC gates pass.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING)
# Combine the conflict-free BLAS K-loop LDS layout with a vectorized zero-LDS
# output path. The scalar output-direct probe was correct but issued four times
# as many VMEM writes; vector-4 is the first gate toward the BLAS writeback
# contract.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_output_v4
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_TILE_DIRECT_STORE
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_TILE_DIRECT_VECTOR4
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Retain CShuffle's proven vectorized global stores, but XOR-swizzle the
# 4-element LDS vector slots by row so its transpose tail no longer aliases a
# small subset of banks.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_VECTOR4)
# Use CK Tile's vector-transaction byte swizzle for both sides of CShuffle.
# This keeps the original ds_write2_b64/ds_read_b128 instruction shape while
# spreading each 16-byte transaction across LDS banks.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_builtin_swizzle
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_BUILTIN_SWIZZLE)
# The producer's template transaction is one BF16 even though LLVM combines
# adjacent stores. Map that scalar offset with the same 16-byte swizzle used by
# the vector load so the combined ds_write2_b64 shape can survive.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_scalar_group16_swizzle
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_SCALAR_GROUP16_SWIZZLE)
# Keep the proven XOR4 descriptor but override the conservative transformed-
# descriptor vector inference for its known contiguous BF16x4 Y6 subspan.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_force_v4_store
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_VECTOR4_FORCE_LDS_STORE)
# Pair the four proven BF16x4 XOR stores as two same-base LDS transactions
# separated by 32 bytes, matching hipBLASLt's ds_write2_b64 offset1:4 tail.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_pair_b64_store
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE)
# Phase165: clean scheduler/cache-locality ablation of the Phase118 winner.
# Keep its compute body and CShuffle byte-for-byte identical while launching
# one workgroup per output tile instead of the 80-block persistent sweep.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_nonpersistent_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_pair_b64_store
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE)
# Phase166: Tensile reports WGM=8 for the selected BLAS solution. CK's existing
# GroupNum=8 is a chiplet-wide permutation, not Tensile's local WGM swizzle.
# Sweep local M01 ownership with GroupNum=1 while keeping Phase118 unchanged.
foreach(tile_m01 IN ITEMS 1 4 8)
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_pair_b64_store_group1_m01_${tile_m01}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
        CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE
        CK_TILE_GROUPED_GEMM_DSREADM_TILE_GROUP_NUM=1
        CK_TILE_GROUPED_GEMM_DSREADM_TILE_M01=${tile_m01})
endforeach()
# Reuse the dead 64-KiB operand allocation for two adjacent M tiles per
# CShuffle pass. This halves the number of epilogue barrier/readback rounds
# while preserving Phase118's XOR descriptor and paired b64 producer.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_pair_b64_store_merge_m2
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE
    CK_TILE_GROUPED_GEMM_CSHUFFLE_MERGE_M2)
# Pipeline the same four 32-KiB slabs through two alternating LDS slots.  This
# lowers the barrier count from eight to five without materializing a doubled
# accumulator tile and therefore targets Phase118's 192-VGPR resource shape.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_pair_b64_store_pipelined_m2
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE
    CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2)
# End each current-slab global store before materializing the next LDS producer
# tile. This preserves the five-barrier schedule while shortening temporary
# register lifetimes enough to target the Phase118 no-spill resource envelope.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_pair_b64_store_pipelined_m2_store_first
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE
    CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2
    CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2_STORE_FIRST)
# Phase-148 launch-structure probe. The device kernel is the validated
# Phase-118 candidate; only an explicit runtime switch splits the argument
# array into one group per non-blocking stream and joins them to the caller
# stream with events. The normal single-stream path remains the default.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_cshuffle_xor4_pair_b64_store_multi_stream
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE
    CK_TILE_GROUPED_GEMM_MULTI_STREAM_EXPERIMENT)
# The vendor DTL1 loop issues all six A/B DSReadM operations for a K16 stage,
# waits once at lgkmcnt(0), then runs the full MMAC group.  Isolate that exact
# synchronous operand schedule from CK's B-only progressive look-ahead while
# retaining the proven BLAS LDS layout and paired CShuffle store.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_blas_lds_sync_cshuffle_xor4_pair_b64_store
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE)
# Reuse the BLAS-style raw-accumulator scalar writeback on the grad-W DSReadM
# body.  This removes CShuffle completely and is isolated behind a dedicated
# mapping gate because the prior validation covered only the RR/grad-X lane.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_direct_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Derive direct scalar coordinates from the raw accumulator tile distribution
# instead of reusing the RR-only bit formula.  This is the correctness-first
# grad-W path toward BLAS's zero-CShuffle buffer_store_short epilogue.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_distributed_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_DISTRIBUTED_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Register-only direct store using the exhaustively enumerated logical output
# distribution. This is the low-resource counterpart of the correct Phase149
# generic-adaptor probe.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_output_mapped_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_MAPPED_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Same exhaustive output mapping, grouped into eight dynamic row bases with
# sixteen immediate-offset scalar stores per base. Exact-tile standalone gate;
# the production selector will retain fallback for padded/tail shapes.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_output_mapped_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_MAPPED_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase167: use hipBLASLt's physical n*2+m wave ownership, concatenate the
# existing DSReadM operand fragments into one 128x64 MIWT8_4 warp GEMM, and
# consume its 128 raw accumulators with the exact BLAS immediate-store map.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_wave_gemm_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase169: same whole-wave compute, but expose C as column-major so the
# contiguous M dimension and 16x8 scalar-store grouping match hipBLASLt.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_wave_gemm_column_major_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase174: verify the same whole-wave/column-major contract for FP16 before
# exposing the transposed grad-W adapter in Primus.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_fp16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_wave_gemm_column_major_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_FP16
    grouped_gemm_fp16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase178: transpose the BF16 R,R grad-X equation into an equivalent
# ColumnMajor x ColumnMajor -> ColumnMajor problem. This tests whether the
# whole-wave compute/contiguous-store contract that closed grad-W also closes
# the remaining grad-X gap without changing the public row-major ABI.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccc_blas_wave_gemm_column_major_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase260: isolate hipBLASLt-style WGM8 traversal on the accepted C,C,C
# body. GroupNum=1 removes CK's outer eight-way remap and M01=8 maps eight
# neighboring M tiles before advancing N, matching the BLAS solution's WGM8
# metadata without changing its producer, consumer, MMAC, or epilogue.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase260_ccc_wgm8
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_STAGE_TILE_GROUP_NUM=1
    CK_TILE_GROUPED_GEMM_DSREADM_STAGE_TILE_M01=8
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase261 traversal ablations: isolate removal of the outer GroupNum=8
# remap while retaining CK's proven M01=4 locality, and independently test
# M01=8 while retaining the outer remap.
foreach(phase261_mapping IN ITEMS g1_m4 g8_m8)
    if(phase261_mapping STREQUAL g1_m4)
        set(phase261_group_num 1)
        set(phase261_m01 4)
    else()
        set(phase261_group_num 8)
        set(phase261_m01 8)
    endif()
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_phase261_ccc_${phase261_mapping}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
        CK_TILE_GROUPED_GEMM_DSREADM_STAGE_TILE_GROUP_NUM=${phase261_group_num}
        CK_TILE_GROUPED_GEMM_DSREADM_STAGE_TILE_M01=${phase261_m01}
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
        CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
        CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
endforeach()
# Phase226: preserve the accepted C,C,C producer/consumer/epilogue contract
# while preventing the compiler from cloning the four-stage hot loop. The
# selected hipBLASLt kernel keeps one compact loop body; this isolates
# instruction-cache pressure without changing any memory or MMAC ordering.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase226_compact_hotloop
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DSREADM_DISABLE_MAIN_LOOP_UNROLL
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase281: the fixed 4x4096x7168x4096 lane always has a hot loop and an even
# K64 tail. Instantiate only that pipeline body so the five dynamic
# TailHandler scenarios do not multiply the selected kernel's text section.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase281_ccc_fixed_hot_even
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DSREADM_FIXED_HOT_EVEN
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase229: keep the K-contiguous C,C,C direct-to-LDS producer but consume
# the resulting tile with the regular DSReadM B mapping used by the faster
# prepacked C,R,C lane.  This is a correctness-first mapping probe: if the
# gfx936 DTL scatter already materializes the regular [K,N] LDS contract, it
# removes the vendor B64 compatibility/permutation path without a global pack.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase229_ccc_standard_b_consumer
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase263: move the vendor B64 fragment permutation from every consumer wave
# to the single producer wave. The hot loop issues the next coalesced B load
# before the current 32-MMAC region and commits the pre-permuted vector after
# it, while the initialization and tail use the same physical LDS contract.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase263_ccc_b_producer_prepermute_overlap
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE_OVERLAP
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase209: preserve the fastest B-only progressive LDS/MMAC overlap while
# consuming the vendor B64 fragment in native order.  This isolates the eight
# per-K16 v_perm instructions without forcing the rejected all-operands-live
# schedule; fixed-m0 remains independently attributable to the accepted lane.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccc_blas_wave_raw_b_progressive_fixed_m0_column_major_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase199: hold the Phase190 C,C -> C compute and epilogue contract fixed,
# but launch one workgroup per output tile. This isolates the persistent
# 80-workgroup scheduler and its repeated group/tile control from the hot loop.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase199_cc_nonpersistent
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_STAGE_NONPERSISTENT
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase201: isolate the output-tile traversal from the Phase190 compute,
# producer, and epilogue contracts. The selected hipBLASLt solution advertises
# WGM8; sweep the two spatial-local remap dimensions around the existing
# GroupNum=8/M01=4 default to test cache reuse on the square 16x16 tile grid.
foreach(phase201_group IN ITEMS 1 8)
    foreach(phase201_m01 IN ITEMS 1 8)
        add_grouped_gemm_fast_target(
            tile_example_grouped_gemm_bf16_phase201_wgm_g${phase201_group}_m${phase201_m01}
            CK_TILE_GROUPED_GEMM_FAST_BF16
            grouped_gemm_bf16.cpp
            CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
            CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
            CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
            CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
            CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
            CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
            CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
            CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
            CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
            CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
            CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
            CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
            CK_TILE_GROUPED_GEMM_DSREADM_TILE_GROUP_NUM=${phase201_group}
            CK_TILE_GROUPED_GEMM_DSREADM_TILE_M01=${phase201_m01}
            CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
            CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
            CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
            CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
    endforeach()
endforeach()
# Phase202: keep Phase190's B-only operand queue and mathematical half4
# contract, but prepare the current B fragment before issuing the current A
# and next-B reads. The resulting lgkmcnt(4) boundary should leave only the
# next B stage outstanding while preserving a contiguous 32-MMAC interval.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase202_b64_early_permute
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase203: Phase202 with the exact same operand lifetimes and ordering, but
# express lgkmcnt(4) through LLVM's waitcnt builtin. This lets the compiler
# retire the current four A reads without adding a redundant 3/2/1/0 ladder.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase203_b64_builtin_wait4
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_BUILTIN_WAIT4
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase204: keep the vendor B64 half4 in native contiguous order and restore
# A/B K agreement at the A consumer packing boundary. Unlike Phase197/198,
# this does not permute global A data or alter the direct-to-LDS producer.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase204_a_consumer_remap_raw_b
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_A_CONSUMER_HALF4_REMAP
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase208: keep Phase202's correct early B-half4 preparation and B-only
# look-ahead, then isolate the cost/benefit of the vendor-style contiguous
# 32-MMAC region. The first target fixes only C; the second additionally
# binds the current A/B fragments to the selected BLAS register banks.
foreach(phase208_variant IN ITEMS fixed_c fixed_ab_c)
    set(phase208_extra_defs CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_C_ASM)
    if(phase208_variant STREQUAL fixed_ab_c)
        list(APPEND phase208_extra_defs
             CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_AB_REGS)
    endif()
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_phase208_early_${phase208_variant}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        ${phase208_extra_defs})
endforeach()

# Phase209: retain Phase202's proven B-only overlap and MMAC path, but emit
# the four A DSReadM operations from the same two address VGPRs with immediate
# offsets 0/128/256/384, matching the selected hipBLASLt consumer ISA.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase209_a_immediate_consumer
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase210: force the Phase209 address pair into one four-instruction asm so
# the three non-zero A displacements are encoded as DSReadM immediates.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase210_a_immediate_asm
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase211: preserve the exact Phase210 DSReadM instructions and dependencies,
# but remove only the compiler-wide memory clobber from that local asm.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase211_a_immediate_asm_relaxed
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_ASM_RELAXED
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase212: keep raw vendor B64 fragments two K16 stages ahead. The compile-time
# A consumer remap preserves correctness without changing coalesced global
# accesses, while the extra B live range targets the raw path's excess LDS
# wait cycles without reintroducing v_perm instructions.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase212_raw_b_lookahead2
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_A_CONSUMER_HALF4_REMAP
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED2
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase214: preserve coalesced global A reads and move the K-half correction
# to the direct-to-LDS destination. This pairs native A DSReadM with raw B64
# without consumer permutations or altered global source addresses.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase214_a_lds_prepermute_raw_b
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_LDS_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase215: keep the next complete A+B K16 operand stage in flight, matching
# hipBLASLt's lgkmcnt(8) cadence, but restore CK's established B half4 order
# only after the current stage has retired.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase215_ab_pipeline_permute
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase216: use the correctness-proven producer-side A K-half permutation so
# both A and B can remain in their native DSReadM order. Keep the same
# two-stage A+B queue and scalar resource recurrence as Phase215, removing the
# eight compatibility v_perm instructions from every K16 stage.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase216_raw_ab_pipeline_recurrence
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase219: combine the correctness-proven raw A/B operand pipeline with the
# vendor N-major accumulator numbering. Restore CK's C fragment contract once
# after the K loop, keeping each hot 32-MMAC region consecutive in v0:v127.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase219_raw_ab_raw_c_restore
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_ORDER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_RESTORE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase217: present all current raw A/B registers to LLVM at one dependency
# boundary so its waitcnt pass can retire the older eight LDS operations
# together instead of inserting lgkmcnt(7/6/5/4) through the first MMACs.
foreach(phase217_variant IN ITEMS fence fence_inline)
    set(phase217_extra_defs
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_FENCE_ALL)
    if(phase217_variant STREQUAL fence_inline)
        list(APPEND phase217_extra_defs
             CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_INLINE_MMAC)
    endif()
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_phase217_raw_ab_${phase217_variant}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
        CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
        CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        ${phase217_extra_defs})
endforeach()

# Phase218: place the complete 32-MMAC region in one relaxed asm dependency
# boundary while leaving physical register selection to LLVM.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase218_relaxed_full_mmac
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase219: retain Phase218's single relaxed 32-MMAC region but return to the
# lower-register B-only look-ahead. The builtin wait retires current A plus
# prior B while leaving the following four B reads outstanding.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase219_b_pipeline_relaxed_full_mmac
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_BUILTIN_WAIT4
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase220: fuse the exact BLAS stage boundary and all 32 MMACs into one asm
# region. This prevents LLVM from hoisting the first consumer above the
# barrier or strengthening the intended lgkmcnt(8) retirement.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase220_fused_sync_ab_mmac
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_AB_MMAC
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase221: lower-register counterpart of Phase220. Fuse lgkmcnt(4),
# vmcnt(4), the block barrier, and 32 MMACs around the B-only look-ahead.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase221_fused_sync_b_mmac
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_B_MMAC
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase222: memory-ordered correction of Phase220. The fused asm keeps its
# exact wait/barrier/MMAC cadence while now forming a compiler memory fence.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase222_fused_sync_ab_mmac_memory
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_AB_MMAC
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase223: keep the correctness-proven fixed-m0/scalar-B Phase178 body, but
# launch one workgroup per output tile. This isolates persistent grid-stride
# control and serial epilogue tails without changing the producer, MMAC, or
# direct-store contracts.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase223_ccc_fixed_m0_scalar_b_nonpersistent
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_STAGE_NONPERSISTENT
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase224: runtime-only queued-grid sweep on the same proven body. Multiplier
# one is the 80-WG control; larger values change only the persistent grid and
# its grid-stride tile assignment.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase224_ccc_fixed_m0_scalar_b_grid_sweep
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase225: retain the selected B look-ahead and add only the first A
# ds_read_m32x16 fragment of the next K16 stage. This shortens the first-A
# dependency without paying the 227-VGPR lifetime of the full A+B pipeline.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase225_ccc_b_a0_pipeline
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_A0_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase226: prepare the current B fragment before issuing the next operand
# reads, communicate lgkmcnt(4) to LLVM, and keep all 32 MMACs in one
# allocator-selected asm region. The external vmcnt/barrier remains in its
# correctness-proven position.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase226_ccc_early_b_builtin_full_mmac
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_BUILTIN_WAIT4
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase227/228: remove B look-ahead entirely so only one A/B K16 operand set
# stays live. This matches the selected hipBLASLt stage topology more closely
# and tests whether the allocator can keep the full 32-MMAC region contiguous
# without Phase219/226's second B fragment. Phase227 keeps the regular MMAC
# lowering as the live-range control; Phase228 changes only the MMAC region to
# the allocator-selected single asm block.
foreach(phase227_variant IN ITEMS regular full_mmac)
    set(phase227_extra_defs)
    if(phase227_variant STREQUAL full_mmac)
        list(APPEND phase227_extra_defs
             CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RELAXED_FULL_MMAC_ASM)
    endif()
    add_grouped_gemm_fast_target(
        tile_example_grouped_gemm_bf16_phase227_ccc_single_stage_${phase227_variant}
        CK_TILE_GROUPED_GEMM_FAST_BF16
        grouped_gemm_bf16.cpp
        CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
        CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
        CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
        CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
        CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
        CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
        CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
        CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
        ${phase227_extra_defs})
endforeach()

# Phase228: keep the correctness-required B look-ahead, consume B64 in native
# order, and express the matching A half4 order only through compile-time
# fragment indices. Unlike A_K16_PREPERMUTE this leaves all global producer
# addresses coalesced. Compose it with the accepted fixed-m0, scalar B resource,
# and scalar-resource recurrence so the v_perm removal is isolated on the
# current fastest body.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase228_ccc_raw_b_a_consumer_remap
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_A_CONSUMER_HALF4_REMAP
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)

# Phase200: consume the vendor ds_read_b64 fragments without CK's half4
# permutation and preserve the selected hipBLASLt solution's native
# [N-repeat][M-repeat][raw-MMAC-register] accumulator numbering through the
# ColumnMajor store. Phase188 changed only B and was therefore incomplete.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase200_vendor_raw_c
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_INLINE_MMAC
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_ORDER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase198: compose the correctness-proven A source K permutation with raw
# B64 and the vendor two-stage A/B operand queue. RunStage retires the older
# eight LDS reads with lgkmcnt(8), leaving the following eight live across
# each 32-MMAC interval.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_phase198_raw_ab_pipeline
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase191: keep four B64 reads for the following K16 stage in flight across
# the current 32-MMAC interval. The current four A reads and prior B fragment
# are retired with lgkmcnt(4); the final K16 stage is explicitly drained.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccc_vendor_b64_wait4_setprio_column_major_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_PIPELINED_WAIT4
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase188: feed the vendor B64 registers to MMAC in their native contiguous
# order. The hardware A and B operand layouts are asymmetric; this isolates
# whether the generic DSReadM [0,2,1,3] compatibility shuffle is unnecessary
# for the exact Ailk/Bijk transposed grad-X contract.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccc_vendor_b64_raw_column_major_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS
    CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND
    CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_C_ASM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_AB_REGS
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_SETPRIO
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase181: correctness/control lane for the transposed grad-X equation.
# Keep the same BLAS-layout wave GEMM and column-major direct epilogue, but
# route C,C operands through CK's regular register-mediated G2L path. This
# isolates the experimental K-contiguous B direct-to-LDS mapping above.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccc_generic_blas_wave_gemm_column_major_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Phase182a: retain the BLAS LDS/wave compute contract but restore CK's
# descriptor-driven CShuffle epilogue for a C,C -> R input-path control.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccr_blas_wave_cshuffle
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_gfx936_bf16_cc_control.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING)
# Phase182b: pure CK DSReadM/CShuffle control for the same C,C -> R problem.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccr_standard_cshuffle
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_gfx936_bf16_cc_control.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING)
# Phase184a/b: separate the BLAS LDS descriptor from the whole-wave MMAC
# operand order for the C,C input path.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccr_blas_lds_standard_wave_cshuffle
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_gfx936_bf16_cc_control.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING)
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_ccr_standard_lds_blas_wave_cshuffle
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_gfx936_bf16_cc_control.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
    CK_TILE_GROUPED_GEMM_DEBUG_16BIT_MAPPING)
# Phase154: pair the proven mapped/immediate direct store with one workgroup
# per output tile. This lets the device overlap store tails from completed
# tiles with compute from later workgroups instead of serializing every tile
# through the 80-block persistent loop.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_nonpersistent_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_output_mapped_immediate_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_STAGE_NONPERSISTENT
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_MAPPED_IMMEDIATE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# No-D grad-W uses set semantics, so test the existing register-to-global
# epilogue against the same compute body. This removes CShuffle LDS traffic
# while leaving multi-D and every default target untouched.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_direct_store
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE)
# True zero-LDS output probe: convert each MMAC warp tile to its logical output
# distribution in registers and store that distributed tile directly.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_output_direct_store
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_TILE_DIRECT_STORE
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# Correctness-first zero-CShuffle scalar-store probe. Unlike the failed raw
# accumulator walk, this traverses MakeCOutputBlockTile's logical [M, N]
# distribution, then writes each BF16 value directly to global memory.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_blas_lds_output_scalar_short
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_TILE_DIRECT_STORE
    CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_TILE_SCALAR_SHORT
    CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
# BLAS grad-W emits its accumulator tiles in a monotonic order. Keep two
# isolated traversal probes to test whether matching that order improves MMAC
# issue/scoreboard behavior without changing the producer or B look-ahead.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_mmac_reverse
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_TRAVERSAL_REVERSE)
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_pipeline_mmac_swap
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
    CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_TRAVERSAL_SWAP)
# Symmetric A-only look-ahead to isolate which operand's LDS latency matters.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_a_pipeline
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_A_OPERAND_PIPELINED)
# B look-ahead plus only the first A fragment of the next K16 stage.
add_grouped_gemm_fast_target(
    tile_example_grouped_gemm_bf16_v3_dsreadm_stage_direct_tn_circular_scalar_resource_b_a0_pipeline
    CK_TILE_GROUPED_GEMM_FAST_BF16
    grouped_gemm_bf16.cpp
    CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
    CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
    CK_TILE_GROUPED_GEMM_DSREADM_B_A0_OPERAND_PIPELINED)
# Default dispatch therefore uses exact NN for both dtypes, staged TN for FP16,
# and exact TN for BF16. Unaligned, small, Split-K and multi-D inputs retain the
# existing V4 fallback. Wider V4 tiles exceeded LDS or transpose constraints;
# scalar LDS scatter was correct but measured 11.54 ms for NN and was replaced
# by the retained b64 producer.

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
    instances/grouped_gemm_gfx936_bf16_tn_direct.cpp
    ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
)
set_source_files_properties(
    instances/grouped_gemm_gfx936_bf16_tn_direct.cpp
    ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
    PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-template;-Wno-unneeded-internal-declaration")
target_include_directories(ck_tile_hcu_grouped_gemm PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ck_tile_hcu_grouped_gemm
    PRIVATE
        CK_TILE_EXAMPLE_NO_MAIN
        CK_TILE_GROUPED_GEMM_FAST_BUILD
        CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
        CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED
        CK_TILE_GROUPED_GEMM_FAST_FP16
        CK_TILE_GROUPED_GEMM_FAST_BF16)
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
    instances/grouped_gemm_gfx936_bf16_tn_direct.cpp
)
set_source_files_properties(
    instances/grouped_gemm_gfx936_bf16_tn_direct.cpp
    PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-template;-Wno-unneeded-internal-declaration")
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


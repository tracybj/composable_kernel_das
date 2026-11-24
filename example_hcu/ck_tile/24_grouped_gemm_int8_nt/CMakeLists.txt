# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
get_filename_component(CK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CKTileGroupedInt8Nt LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(GPU_TARGETS gfx936 CACHE STRING "HCU architecture")
    include_directories(${CK_ROOT}/include ${CK_ROOT}/library/include)
    add_compile_options(-xhip --offload-arch=${GPU_TARGETS})
    function(ck_tile_hcu_grouped_gemm_apply_codegen target_name)
        target_compile_options(${target_name} PRIVATE
            -fno-offload-uniform-block
            "SHELL: -mllvm --lsr-drop-solution=1"
            "SHELL: -mllvm -enable-post-misched=0"
            "SHELL: -mllvm -amdgpu-early-inline-all=true"
            -fno-gpu-rdc)
    endfunction()
endif()
# The full CK library also compiles instances.cpp through its provider manifest.
if(TARGET ck_tile_hcu_grouped_gemm AND GPU_TARGETS MATCHES "gfx936")
    target_compile_definitions(ck_tile_hcu_grouped_gemm PRIVATE
        CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_INT8_NT)
    target_compile_definitions(ck_tile_hcu_grouped_gemm_device_args_providers PRIVATE
        CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION)
endif()

set(CK_INT8_BENCHMARK_HIPBLASLT_ROOT "" CACHE PATH
    "hipBLASLt installation for the optional Example 19 raw INT8 benchmark")
set(INT8_BLAS_ROOT "${CK_INT8_BENCHMARK_HIPBLASLT_ROOT}")
if(INT8_BLAS_ROOT AND TARGET tile_example_grouped_gemm_device_args_benchmark)
    add_library(ck_tile_grouped_int8_nt_benchmark_support STATIC EXCLUDE_FROM_ALL
        benchmarks/public_bridge.cpp benchmarks/int8_benchmark.cpp benchmarks/blas_bridge.cpp)
    target_compile_options(ck_tile_grouped_int8_nt_benchmark_support PRIVATE
        -Wno-global-constructors -Wno-sign-compare)
    target_include_directories(ck_tile_grouped_int8_nt_benchmark_support PRIVATE
        ${INT8_BLAS_ROOT}/include)
    target_compile_definitions(ck_tile_grouped_int8_nt_benchmark_support PRIVATE
        CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION)
    target_link_libraries(ck_tile_grouped_int8_nt_benchmark_support PUBLIC
        ${INT8_BLAS_ROOT}/lib/libhipblaslt.so.0.10 dl)

    target_compile_definitions(tile_example_grouped_gemm_device_args_benchmark PRIVATE
        CK_TILE_GROUPED_GEMM_HAS_INT8_BENCHMARK)
    target_link_libraries(tile_example_grouped_gemm_device_args_benchmark PRIVATE
        ck_tile_grouped_int8_nt_benchmark_support)
endif()

# ---------------------------------------------------------------------------
# INT8 NT Cached Weights and Fused Scale Extension API
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
option(CK_INT8_CACHED_TUNING "Build the offline scaled INT8 NT tuning API" OFF)

# The provider archive owns the HIP launches used by both INT8 NT APIs.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_int8_nt_device_args_provider_sources.txt"
    INT8_NT_DEVICE_ARGS_PROVIDER_SOURCES REGEX "^[^#].+")
set_source_files_properties(${INT8_NT_DEVICE_ARGS_PROVIDER_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers STATIC EXCLUDE_FROM_ALL
    ${INT8_NT_DEVICE_ARGS_PROVIDER_SOURCES})
target_include_directories(ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers PRIVATE
    ${CK_ROOT}/include ${CK_ROOT}/library/include)
target_compile_features(ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers PRIVATE cxx_std_17)
target_compile_definitions(ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers PRIVATE
    CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION
    $<$<BOOL:${CK_INT8_CACHED_TUNING}>:CK_INT8_CACHED_TUNING>)
target_compile_options(ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers PRIVATE
    -Wno-global-constructors -Wno-switch-enum -Wno-undefined-reinterpret-cast
    -Wno-sign-compare -Wno-unused-parameter
    "SHELL: -mllvm -support-768-vgprs=true")
set_target_properties(ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers)

add_library(ck_tile_int8_nt_cached SHARED grouped_int8_nt_cached_api.cpp)
target_link_libraries(ck_tile_int8_nt_cached PRIVATE
    Threads::Threads ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers)
target_include_directories(ck_tile_int8_nt_cached PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CK_ROOT}/include
    ${CK_ROOT}/library/include)
target_compile_features(ck_tile_int8_nt_cached PRIVATE cxx_std_17)
target_compile_options(ck_tile_int8_nt_cached PRIVATE
    -Wno-global-constructors -Wno-switch-enum -Wno-undefined-reinterpret-cast
    -Wno-sign-compare -Wno-unused-parameter
    "SHELL: -mllvm -support-768-vgprs=true")
target_compile_definitions(ck_tile_int8_nt_cached PRIVATE
    CK_ENABLE_INT8
    CK_EXPERIMENTAL_BIT_INT_EXTENSION
    $<$<BOOL:${CK_INT8_CACHED_TUNING}>:CK_INT8_CACHED_TUNING>)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_int8_nt_cached)
set_target_properties(ck_tile_int8_nt_cached PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

add_library(ck_tile_int8_nt_cached_benchmark SHARED EXCLUDE_FROM_ALL
    benchmarks/benchmark_bridge.cpp)
target_link_libraries(ck_tile_int8_nt_cached_benchmark PRIVATE ck_tile_int8_nt_cached)
target_include_directories(ck_tile_int8_nt_cached_benchmark PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CK_ROOT}/include)
target_compile_features(ck_tile_int8_nt_cached_benchmark PRIVATE cxx_std_17)
target_compile_options(ck_tile_int8_nt_cached_benchmark PRIVATE
    -Wno-global-constructors -Wno-switch-enum)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_int8_nt_cached_benchmark)

if(TARGET tile_example_grouped_gemm_device_args_benchmark AND GPU_TARGETS MATCHES "gfx936")
    target_compile_definitions(tile_example_grouped_gemm_device_args_benchmark PRIVATE
        CK_TILE_GROUPED_GEMM_HAS_SCALED_INT8)
    target_link_libraries(tile_example_grouped_gemm_device_args_benchmark PRIVATE
        ck_tile_int8_nt_cached)
endif()

# Example 24 owns the scaled device-args API and its provider manifest.
if(GPU_TARGETS MATCHES "gfx936")
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        enable_testing()
    endif()

    add_library(ck_tile_hcu_grouped_gemm_int8_nt_device_args STATIC EXCLUDE_FROM_ALL
        grouped_gemm_int8_nt_device_args.cpp)
    target_link_libraries(ck_tile_hcu_grouped_gemm_int8_nt_device_args
        PUBLIC ck_tile_hcu_grouped_gemm_int8_nt_device_args_providers)
    target_include_directories(ck_tile_hcu_grouped_gemm_int8_nt_device_args
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include ${CK_ROOT}/include)
    target_compile_features(ck_tile_hcu_grouped_gemm_int8_nt_device_args PUBLIC cxx_std_17)
    target_compile_definitions(ck_tile_hcu_grouped_gemm_int8_nt_device_args PRIVATE
        CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION)
    target_compile_options(ck_tile_hcu_grouped_gemm_int8_nt_device_args PRIVATE
        -Wno-switch-enum -Wno-undefined-reinterpret-cast -Wno-sign-compare -Wno-unused-parameter
        "SHELL: -mllvm -support-768-vgprs=true")
    set_target_properties(ck_tile_hcu_grouped_gemm_int8_nt_device_args PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_hcu_grouped_gemm_int8_nt_device_args)

    add_executable(tile_example_grouped_gemm_int8_nt_device_args EXCLUDE_FROM_ALL
        benchmarks/device_args_benchmark.cpp)
    target_link_libraries(tile_example_grouped_gemm_int8_nt_device_args PRIVATE
        ck_tile_hcu_grouped_gemm_int8_nt_device_args)

    function(add_int8_nt_device_args_smoke flavor)
        add_test(NAME tile_example_grouped_gemm_int8_nt_device_args_${flavor}
            COMMAND $<TARGET_FILE:tile_example_grouped_gemm_int8_nt_device_args>
                    ${ARGN} --time=0 --correctness-repeats=1)
        set_tests_properties(tile_example_grouped_gemm_int8_nt_device_args_${flavor}
            PROPERTIES LABELS "hcu;ck_tile;int8_nt;device_args;smoke")
    endfunction()

    add_int8_nt_device_args_smoke(raw_bf16
        --m=128 --n=2048 --k=1024 --groups=4 --dtype=int8_bf16)
    add_int8_nt_device_args_smoke(ragged_fp16
        --m=65 --n=256 --k=512 --dtype=int8_fp16 --group-lengths=1,0,65
        --variable-capacity=128 --allow-empty-groups=1 --include-args-build=1)
    add_int8_nt_device_args_smoke(kparallel_bf16
        --m=1 --n=4096 --k=6144 --groups=8 --dtype=int8_bf16)
    add_int8_nt_device_args_smoke(kparallel_fp16
        --m=2 --n=4096 --k=7168 --groups=8 --dtype=int8_fp16)

    add_custom_target(tile_example_grouped_gemm_int8_nt_device_args_c_header_smoke
        COMMAND ${CMAKE_C_COMPILER}
                -std=c11
                -Wall
                -Wextra
                -Werror
                -I${CK_ROOT}/include
                -I${CMAKE_CURRENT_LIST_DIR}/include
                -fsyntax-only
                ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_int8_nt_device_args_c_header_smoke.c
        COMMENT "Compiling INT8 NT grouped-GEMM device-args v2 header as pure C11"
        VERBATIM)
    add_test(NAME tile_example_grouped_gemm_int8_nt_device_args_c_header_smoke
        COMMAND ${CMAKE_C_COMPILER}
                -std=c11
                -Wall
                -Wextra
                -Werror
                -I${CK_ROOT}/include
                -I${CMAKE_CURRENT_LIST_DIR}/include
                -fsyntax-only
                ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_int8_nt_device_args_c_header_smoke.c)
    set_tests_properties(tile_example_grouped_gemm_int8_nt_device_args_c_header_smoke
        PROPERTIES LABELS "hcu;ck_tile;int8_nt;device_args;smoke")
endif()

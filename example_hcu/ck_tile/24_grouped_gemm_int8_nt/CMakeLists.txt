# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
get_filename_component(CK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CKTileGroupedInt8Nt LANGUAGES CXX)
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

# Small development archive with the same public ABI, restricted to INT8 NT.
# Link this OR the full CK grouped-GEMM library, not both ABI owners.
add_library(ck_tile_hcu_grouped_gemm_int8_nt STATIC EXCLUDE_FROM_ALL
    instances.cpp ../19_grouped_gemm/grouped_gemm_device_args.cpp)
target_compile_definitions(ck_tile_hcu_grouped_gemm_int8_nt PRIVATE
    CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION
    CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_INT8_NT
    CK_TILE_GROUPED_GEMM_INT8_NT_ONLY)
target_compile_options(ck_tile_hcu_grouped_gemm_int8_nt PRIVATE
    -Wno-switch-enum -Wno-undefined-reinterpret-cast -Wno-sign-compare -Wno-unused-parameter
    "SHELL: -mllvm -support-768-vgprs=true")
target_include_directories(ck_tile_hcu_grouped_gemm_int8_nt PRIVATE
    ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../19_grouped_gemm)
set_target_properties(ck_tile_hcu_grouped_gemm_int8_nt PROPERTIES POSITION_INDEPENDENT_CODE ON)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_hcu_grouped_gemm_int8_nt)

add_library(ck_tile_grouped_int8_nt_benchmark_api SHARED EXCLUDE_FROM_ALL benchmarks/public_bridge.cpp)
target_link_libraries(ck_tile_grouped_int8_nt_benchmark_api PRIVATE ck_tile_hcu_grouped_gemm_int8_nt)
target_compile_definitions(ck_tile_grouped_int8_nt_benchmark_api PRIVATE
    CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION)
set_target_properties(ck_tile_grouped_int8_nt_benchmark_api PROPERTIES
    OUTPUT_NAME int8_nt_public LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(tile_example_grouped_gemm_device_args_benchmark_int8 EXCLUDE_FROM_ALL
        ../19_grouped_gemm/grouped_gemm_device_args_benchmark.cpp)
    target_include_directories(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
        /opt/dtk/hipblas/include /opt/dtk/include)
    target_link_libraries(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
        ck_tile_hcu_grouped_gemm_int8_nt ${CMAKE_DL_LIBS})
endif()

set(CK_INT8_BENCHMARK_HIPBLASLT_ROOT "" CACHE PATH
    "hipBLASLt installation for the optional raw INT8 comparison benchmark")
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CK_INT8_BENCHMARK_HIPBLASLT_ROOT)
    set(CK_INT8_BENCHMARK_HIPBLASLT_ROOT "${HIPBLASLT_INSTALL}")
endif()
set(INT8_BLAS_ROOT "${CK_INT8_BENCHMARK_HIPBLASLT_ROOT}")
if(INT8_BLAS_ROOT)
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

    add_executable(tile_example_grouped_gemm_int8_nt EXCLUDE_FROM_ALL benchmarks/int8_benchmark_main.cpp)
    target_link_libraries(tile_example_grouped_gemm_int8_nt PRIVATE
        ck_tile_grouped_int8_nt_benchmark_support ck_tile_hcu_grouped_gemm_int8_nt)
    set_target_properties(tile_example_grouped_gemm_int8_nt PROPERTIES
        BUILD_RPATH "${INT8_BLAS_ROOT}/lib;/opt/dtk/hip/lib")

    if(TARGET tile_example_grouped_gemm_device_args_benchmark_int8)
        target_compile_definitions(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
            CK_TILE_GROUPED_GEMM_HAS_INT8_BENCHMARK CK_EXPERIMENTAL_BIT_INT_EXTENSION)
        target_include_directories(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
            ${INT8_BLAS_ROOT}/include)
        target_link_libraries(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
            ck_tile_grouped_int8_nt_benchmark_support ck_tile_hcu_grouped_gemm_int8_nt)
        set_target_properties(tile_example_grouped_gemm_device_args_benchmark_int8 PROPERTIES
            BUILD_RPATH "${INT8_BLAS_ROOT}/lib;/opt/dtk/hip/lib")
    endif()

    if(TARGET tile_example_grouped_gemm_device_args_benchmark)
        target_compile_definitions(tile_example_grouped_gemm_device_args_benchmark PRIVATE
            CK_TILE_GROUPED_GEMM_HAS_INT8_BENCHMARK)
        target_link_libraries(tile_example_grouped_gemm_device_args_benchmark PRIVATE
            ck_tile_grouped_int8_nt_benchmark_support)
    endif()
endif()

# ---------------------------------------------------------------------------
# INT8 NT Cached Weights and Fused Scale Extension API
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
option(CK_INT8_CACHED_TUNING "Build the offline scaled INT8 NT tuning API" OFF)

add_library(ck_tile_int8_nt_cached SHARED
    grouped_int8_nt_cached_api.cpp
    instances.cpp
    instances/grouped_int8_nt_cached_dispatch.cpp
    instances/grouped_int8_nt_scaled_dispatch.cpp)
target_link_libraries(ck_tile_int8_nt_cached PRIVATE Threads::Threads)
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

foreach(benchmark IN ITEMS tile_example_grouped_gemm_device_args_benchmark
                           tile_example_grouped_gemm_device_args_benchmark_int8)
    if(TARGET ${benchmark} AND GPU_TARGETS MATCHES "gfx936")
        target_compile_definitions(${benchmark} PRIVATE CK_TILE_GROUPED_GEMM_HAS_SCALED_INT8)
        target_link_libraries(${benchmark} PRIVATE ck_tile_int8_nt_cached)
    endif()
endforeach()

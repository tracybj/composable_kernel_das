# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CKTileGroupedInt8Nt LANGUAGES CXX)
    get_filename_component(CK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    set(CMAKE_CXX_STANDARD 17)
    set(GPU_TARGETS gfx936 CACHE STRING "HCU architecture")
    include_directories(${CK_ROOT}/include ${CK_ROOT}/library/include)
    add_compile_options(-xhip --offload-arch=${GPU_TARGETS}
        -mllvm -support-768-vgprs=true)
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
target_include_directories(ck_tile_hcu_grouped_gemm_int8_nt PRIVATE
    ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../19_grouped_gemm)
set_target_properties(ck_tile_hcu_grouped_gemm_int8_nt PROPERTIES POSITION_INDEPENDENT_CODE ON)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_hcu_grouped_gemm_int8_nt)

add_library(ck_tile_grouped_int8_nt_benchmark_api SHARED EXCLUDE_FROM_ALL public_bridge.cpp)
target_link_libraries(ck_tile_grouped_int8_nt_benchmark_api PRIVATE ck_tile_hcu_grouped_gemm_int8_nt)
target_compile_definitions(ck_tile_grouped_int8_nt_benchmark_api PRIVATE
    CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION)
set_target_properties(ck_tile_grouped_int8_nt_benchmark_api PROPERTIES
    OUTPUT_NAME int8_nt_public LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(HIPBLASLT_INSTALL "" CACHE PATH "INT8 reference hipBLASLt installation")
if(HIPBLASLT_INSTALL)
    add_library(ck_tile_grouped_int8_nt_benchmark_support STATIC EXCLUDE_FROM_ALL
        public_bridge.cpp int8_benchmark.cpp blas_bridge.cpp)
    target_include_directories(ck_tile_grouped_int8_nt_benchmark_support PRIVATE
        ${HIPBLASLT_INSTALL}/include)
    target_compile_definitions(ck_tile_grouped_int8_nt_benchmark_support PRIVATE
        CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION)
    target_link_libraries(ck_tile_grouped_int8_nt_benchmark_support PUBLIC
        ${HIPBLASLT_INSTALL}/lib/libhipblaslt.so.0.10 dl)

    add_executable(tile_example_grouped_gemm_int8_nt EXCLUDE_FROM_ALL int8_benchmark_main.cpp)
    target_link_libraries(tile_example_grouped_gemm_int8_nt PRIVATE
        ck_tile_grouped_int8_nt_benchmark_support ck_tile_hcu_grouped_gemm_int8_nt)
    set_target_properties(tile_example_grouped_gemm_int8_nt PROPERTIES
        BUILD_RPATH "${HIPBLASLT_INSTALL}/lib;/opt/dtk/hip/lib")

    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        add_executable(tile_example_grouped_gemm_device_args_benchmark_int8 EXCLUDE_FROM_ALL
            ../19_grouped_gemm/grouped_gemm_device_args_benchmark.cpp)
        target_compile_definitions(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
            CK_TILE_GROUPED_GEMM_HAS_INT8_BENCHMARK CK_EXPERIMENTAL_BIT_INT_EXTENSION)
        target_include_directories(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
            ${HIPBLASLT_INSTALL}/include /opt/dtk/hipblas/include /opt/dtk/include)
        target_link_libraries(tile_example_grouped_gemm_device_args_benchmark_int8 PRIVATE
            ck_tile_grouped_int8_nt_benchmark_support ck_tile_hcu_grouped_gemm_int8_nt)
        set_target_properties(tile_example_grouped_gemm_device_args_benchmark_int8 PROPERTIES
            BUILD_RPATH "${HIPBLASLT_INSTALL}/lib;/opt/dtk/hip/lib")
    endif()

    if(TARGET tile_example_grouped_gemm_device_args_benchmark)
        target_compile_definitions(tile_example_grouped_gemm_device_args_benchmark PRIVATE
            CK_TILE_GROUPED_GEMM_HAS_INT8_BENCHMARK)
        target_link_libraries(tile_example_grouped_gemm_device_args_benchmark PRIVATE
            ck_tile_grouped_int8_nt_benchmark_support)
    endif()
endif()

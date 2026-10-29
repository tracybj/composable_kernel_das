# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
 project(CKTileInt8CachedWeights LANGUAGES CXX)
endif()
get_filename_component(CK_INT8_CACHED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(CK_INT8_CACHED_ARCH gfx936 CACHE STRING "Architecture for the INT8-only extension")
add_library(ck_tile_int8_nt_cached SHARED cached_api.cpp cached_dispatch.cpp)
find_package(Threads REQUIRED)
target_link_libraries(ck_tile_int8_nt_cached PRIVATE Threads::Threads)
target_include_directories(ck_tile_int8_nt_cached PRIVATE
 ${CK_INT8_CACHED_ROOT}/include ${CK_INT8_CACHED_ROOT}/library/include)
target_compile_features(ck_tile_int8_nt_cached PRIVATE cxx_std_17)
target_compile_definitions(ck_tile_int8_nt_cached PRIVATE CK_ENABLE_INT8 CK_EXPERIMENTAL_BIT_INT_EXTENSION)
target_compile_options(ck_tile_int8_nt_cached PRIVATE -xhip --offload-arch=${CK_INT8_CACHED_ARCH}
 "SHELL: -mllvm -support-768-vgprs=true" -fno-offload-uniform-block
 "SHELL: -mllvm --lsr-drop-solution=1" "SHELL: -mllvm -enable-post-misched=0"
 "SHELL: -mllvm -amdgpu-early-inline-all=true" -fno-gpu-rdc)
set_target_properties(ck_tile_int8_nt_cached PROPERTIES
 POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
# This extension defines no old grouped-GEMM ABI owner and links no FP16/BF16
# provider. Its symbols and INT8 dispatch are private except the new C API.
add_library(ck_tile_int8_nt_cached_benchmark SHARED EXCLUDE_FROM_ALL benchmark_bridge.cpp)
target_link_libraries(ck_tile_int8_nt_cached_benchmark PRIVATE ck_tile_int8_nt_cached)
target_compile_options(ck_tile_int8_nt_cached_benchmark PRIVATE -xhip --offload-arch=${CK_INT8_CACHED_ARCH})
target_compile_features(ck_tile_int8_nt_cached_benchmark PRIVATE cxx_std_17)

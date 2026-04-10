set(TILE_EXAMPLE_GROUPED_GEMM "tile_example_grouped_gemm")
message("adding ${TILE_EXAMPLE_GROUPED_GEMM}")
add_compile_options("SHELL: -mllvm -support-768-vgprs=true -Rpass-analysis=kernel-resource-usage")
add_executable(${TILE_EXAMPLE_GROUPED_GEMM} grouped_gemm.cpp)

# add_executable(tile_example_quant_grouped_gemm EXCLUDE_FROM_ALL quant_grouped_gemm.cpp)
# add_executable(tile_example_grouped_gemm_preshuffle EXCLUDE_FROM_ALL grouped_gemm_preshuffle.cpp)

add_compile_options("SHELL: -mllvm -support-768-vgprs=true -Rpass-analysis=kernel-resource-usage")
add_example_executable(tile_example_conv3d_fwd  conv3d_fwd.cpp)

# hcu batched gemm examples
add_custom_target(example_hcu_batched_gemm_xdl)

add_example_executable(example_hcu_batched_gemm_xdl_bf16_v3 batched_gemm_xdl_bf16_v3.cpp)

add_dependencies(example_hcu_batched_gemm_xdl
                 example_hcu_batched_gemm_xdl_bf16_v3)

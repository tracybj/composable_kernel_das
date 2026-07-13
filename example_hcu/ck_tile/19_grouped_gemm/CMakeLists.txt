set(TILE_EXAMPLE_GROUPED_GEMM "tile_example_grouped_gemm")
message("adding ${TILE_EXAMPLE_GROUPED_GEMM}")

file(GLOB INSTANCE_SRCS instances/grouped_gemm_*.cpp)

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
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp8_fast  CK_TILE_GROUPED_GEMM_FAST_FP8  grouped_gemm_fp8.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf8_fast  CK_TILE_GROUPED_GEMM_FAST_BF8  grouped_gemm_bf8.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_int8_fast CK_TILE_GROUPED_GEMM_FAST_INT8 grouped_gemm_int8.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_int4_fast CK_TILE_GROUPED_GEMM_FAST_INT4 grouped_gemm_int4.cpp)

add_library(ck_tile_dcu_grouped_gemm STATIC EXCLUDE_FROM_ALL
    grouped_gemm.cpp
    instances/grouped_gemm_fp16.cpp
    instances/grouped_gemm_bf16.cpp
)
target_include_directories(ck_tile_dcu_grouped_gemm PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ck_tile_dcu_grouped_gemm
    PRIVATE
        CK_TILE_EXAMPLE_NO_MAIN
        CK_TILE_GROUPED_GEMM_FAST_BUILD
        CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
        CK_TILE_GROUPED_GEMM_FAST_FP16
        CK_TILE_GROUPED_GEMM_FAST_BF16)
set_target_properties(ck_tile_dcu_grouped_gemm PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tile_example_grouped_gemm_c_api_smoke EXCLUDE_FROM_ALL
    grouped_gemm_c_api_smoke.cpp
)
target_link_libraries(tile_example_grouped_gemm_c_api_smoke PRIVATE ck_tile_dcu_grouped_gemm)

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

function(add_grouped_gemm_fast_smoke_test test_name target_name precision b_layout shape_args)
    add_test(NAME ${test_name}
        COMMAND $<TARGET_FILE:${target_name}>
                -prec=${precision}
                -a_layout=R
                -b_layout=${b_layout}
                ${${shape_args}}
    )
    set_tests_properties(${test_name}
        PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;smoke"
    )
endfunction()

add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nt_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 C GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nt_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 C GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS)

add_test(NAME tile_example_grouped_gemm_c_api_nn_variable_k_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_c_api_smoke>
)
set_tests_properties(tile_example_grouped_gemm_c_api_nn_variable_k_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;c_api;smoke"
)


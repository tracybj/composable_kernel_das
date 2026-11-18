// SPDX-License-Identifier: MIT

#include "grouped_gemm_int8_nt_device_args.h"

int ck_tile_hcu_grouped_gemm_int8_nt_device_args_c_header_smoke(void)
{
    ck_tile_hcu_grouped_gemm_int8_problem_v2 problem = {0};
    problem.struct_size                              = (uint32_t)sizeof(problem);
    problem.abi_version                              = CK_TILE_HCU_GROUPED_GEMM_SCALED_ABI_VERSION;
    return problem.struct_size == 0u ||
           sizeof(ck_tile_hcu_grouped_gemm_int8_scale_v2) == 0u ||
           sizeof(ck_tile_hcu_grouped_gemm_int8_launch_v2) == 0u ||
           sizeof(ck_tile_hcu_grouped_gemm_int8_candidate_info_v2) == 0u ||
           sizeof(ck_tile_hcu_grouped_gemm_int8_pack_arg_v2) == 0u;
}

// SPDX-License-Identifier: MIT
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include <cstdlib>
#include <iostream>
int main()
{
    using namespace ck_tile;
    GroupedGemmProblem p{};
    p.arch        = GroupedGemmArch::gfx936;
    p.data_type   = GroupedGemmDataType::int8_int32;
    p.layout      = GroupedGemmLayout::nt;
    p.group_count = 8;
    p.m           = GroupedGemmDimensionSummary::Common(256);
    p.n           = GroupedGemmDimensionSummary::Common(4096);
    p.k           = GroupedGemmDimensionSummary::Common(4096);
    int checks    = 0;
    auto expect   = [&](GroupedGemmInstanceId want) {
        ++checks;
        const auto got = select_grouped_gemm_candidate(p);
        if(got.instance_id != want)
        {
            std::cerr << "selector contract failure " << checks << std::endl;
            std::exit(1);
        }
    };
    expect(GroupedGemmInstanceId::int8_nt_m256_n256_aligned);
    p.may_have_empty_groups = true;
    expect(GroupedGemmInstanceId::int8_nt_m128);
    p.may_have_empty_groups    = false;
    p.group_lengths_are_ragged = true;
    expect(GroupedGemmInstanceId::int8_nt_m128);
    p.group_lengths_are_ragged = false;
    p.m                        = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(2048);
    expect(GroupedGemmInstanceId::int8_nt_m128);
    p.m = GroupedGemmDimensionSummary::Common(256);
    p.n = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(4096);
    expect(GroupedGemmInstanceId::none);
    p.n = GroupedGemmDimensionSummary::Common(4096);
    p.k = GroupedGemmDimensionSummary::Common(4112);
    expect(GroupedGemmInstanceId::int8_nt_m256_n256_padded);
    p.k = GroupedGemmDimensionSummary::Common(512);
    expect(GroupedGemmInstanceId::int8_nt_m128_aligned);
    p.m = GroupedGemmDimensionSummary::Common(16);
    expect(GroupedGemmInstanceId::int8_nt_small_m_n64_k128);
    p.m = GroupedGemmDimensionSummary::Common(32);
    expect(GroupedGemmInstanceId::int8_nt_m32);
    p.m = GroupedGemmDimensionSummary::Common(64);
    expect(GroupedGemmInstanceId::int8_nt_m64);
    p.k = GroupedGemmDimensionSummary::Common(131072);
    expect(GroupedGemmInstanceId::none);
    p.k    = GroupedGemmDimensionSummary::Common(512);
    p.arch = GroupedGemmArch::gfx938;
    expect(GroupedGemmInstanceId::none);
    p.arch   = GroupedGemmArch::gfx936;
    p.layout = GroupedGemmLayout::nn;
    expect(GroupedGemmInstanceId::none);
    p.layout = GroupedGemmLayout::nt;
    p.m      = GroupedGemmDimensionSummary::Common(16384);
    p.k      = GroupedGemmDimensionSummary::Common(4096);
    expect(GroupedGemmInstanceId::int8_nt_m256_n256_ngroup);
    p.k = GroupedGemmDimensionSummary::Common(2048);
    expect(GroupedGemmInstanceId::int8_nt_m128_mgroup8);
    p.k = GroupedGemmDimensionSummary::Common(4096);
    p.m = GroupedGemmDimensionSummary::Common(4224);
    expect(GroupedGemmInstanceId::int8_nt_m128_n256_aligned);
    p.n = GroupedGemmDimensionSummary::Common(4224);
    expect(GroupedGemmInstanceId::int8_nt_m128);
    p.m = GroupedGemmDimensionSummary::Common(16384);
    p.k = GroupedGemmDimensionSummary::Common(1232);
    p.n = GroupedGemmDimensionSummary::Common(3072);
    expect(GroupedGemmInstanceId::int8_nt_m128_padded_mgroup8);
    p.m = GroupedGemmDimensionSummary::Common(16);
    p.n = GroupedGemmDimensionSummary::Common(4096);
    p.k = GroupedGemmDimensionSummary::Common(7168);
    expect(GroupedGemmInstanceId::int8_nt_small_g_n32_k512);
    p.k = GroupedGemmDimensionSummary::Common(4080);
    expect(GroupedGemmInstanceId::int8_nt_small_m_n64_k256);
    p.k = GroupedGemmDimensionSummary::Common(4096);
    expect(GroupedGemmInstanceId::int8_nt_small_g_n32_k512);
    p.group_count = 9;
    expect(GroupedGemmInstanceId::int8_nt_small_m_n64_k256);
    p.group_count = 8;
    p.k = GroupedGemmDimensionSummary::Common(7168);
    p.group_count = 32;
    expect(GroupedGemmInstanceId::int8_nt_small_m_n128_k256);
    p.group_count = 8;
    p.may_have_empty_groups = true;
    expect(GroupedGemmInstanceId::int8_nt_direct_m16);
    std::cout << "INT8_SELECTOR_PASS checks=" << checks << std::endl;
}

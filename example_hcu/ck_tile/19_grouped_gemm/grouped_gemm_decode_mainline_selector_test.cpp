// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <iostream>
#include <vector>

int main()
{
    using namespace ck_tile;
    GroupedGemmSelectorPolicy policy{};
    policy.enable_bf16_v7_nn = true;
    const auto original = policy;
    bool pass = !policy.enable_bf16_nn_decode_mainline && !policy.enable_bf16_nn_decode_short;
    int checks = 0;
    const auto main_expected = [](int variant, int m, std::int64_t k) {
        if(variant >= 13 && variant <= 16 && m < 17 && k < 6144) return 0;
        if(variant == 12 || variant == 14) return 6023;
        if(variant == 13) return 6022;
        if(variant == 15) return 6013;
        if(variant == 16) return 6015;
        if(variant == 18 && m < 17 && k < 6144) return 0;
        if(variant == 19 && m < 17) return 0;
        if(variant == 17 || variant == 18) return 6024;
        if(variant == 19) return 6025;
        if(variant == 20) return m <= 16 ? 6024 : 6025;
        return variant >= 9 ? 6020 + variant - 9 : variant >= 6 ? 6013 + variant - 6 :
               variant >= 3 ? 6009 + variant - 3 : 6004 + variant;
    };
    auto check = [&](const GroupedGemmProblem& p, int expected) {
        const auto got = select_grouped_gemm_candidate(p, policy);
        const auto old = select_grouped_gemm_candidate(p, original);
        ++checks;
        const bool ok = expected ? static_cast<int>(got.instance_id) == expected :
            got.instance_id == old.instance_id && got.reason == old.reason;
        if(!ok) std::cerr << "DECODE_MAINLINE_SELECTOR_FAIL case=" << checks
                         << " expected=" << expected << " actual=" << int(got.instance_id) << '\n';
        return ok;
    };
    GroupedGemmProblem p{};
    p.arch = GroupedGemmArch::gfx936;
    p.data_type = GroupedGemmDataType::bf16;
    p.layout = GroupedGemmLayout::nn;
    p.group_count = 32;
    p.n = GroupedGemmDimensionSummary::Common(1024);
    p.k = GroupedGemmDimensionSummary::Common(2048);
    for(bool short_path : {false, true})
    {
        policy.enable_bf16_nn_decode_mainline = !short_path;
        policy.enable_bf16_nn_decode_short = short_path;
        p.k.extent = short_path ? 768 : 2048;
        for(int variant = 0; variant < (short_path ? 6 : 21); ++variant)
        {
            policy.bf16_nn_decode_mainline_variant = variant;
            policy.bf16_nn_decode_short_variant = variant;
            for(int m = 1; m <= 64; ++m)
            {
                p.m = GroupedGemmDimensionSummary::Common(m);
                const int expected = short_path ? (variant >= 3 ? 6016 + variant - 3 : variant == 2 ? 6012 : 6007 + variant) :
                    main_expected(variant, m, p.k.extent);
                pass &= check(p, expected);
            }
        }
        policy.bf16_nn_decode_mainline_variant = 0;
        policy.bf16_nn_decode_short_variant = 0;
        std::vector<GroupedGemmProblem> rejected;
        for(auto arch : {GroupedGemmArch::gfx938, GroupedGemmArch::gfx946, GroupedGemmArch::unsupported})
        { auto x = p; x.arch = arch; rejected.push_back(x); }
        { auto x = p; x.data_type = GroupedGemmDataType::fp16; rejected.push_back(x); }
        for(auto layout : {GroupedGemmLayout::nt, GroupedGemmLayout::tn})
        { auto x = p; x.layout = layout; rejected.push_back(x); }
        for(int m : {0, 65, 128, 256, 2048})
        { auto x = p; x.m.extent = m; rejected.push_back(x); }
        for(int n : {0, 127, 1232, 4097, 8192})
        { auto x = p; x.n.extent = n; rejected.push_back(x); }
        for(int k : (short_path ? std::vector<int>{0, 32, 63, 65, 2047, 2048, 4096} :
                                  std::vector<int>{0, 64, 1984, 2049, 2112, 8193, 16384}))
        { auto x = p; x.k.extent = k; rejected.push_back(x); }
        for(int g : (short_path ? std::vector<int>{0, 33, 1024, 1025} : std::vector<int>{0, 1025}))
        { auto x = p; x.group_count = g; rejected.push_back(x); }
        { auto x = p; x.m = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(64 * 32); rejected.push_back(x); }
        { auto x = p; x.k = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(p.k.extent * 32); rejected.push_back(x); }
        { auto x = p; x.group_lengths_are_ragged = true; rejected.push_back(x); }
        { auto x = p; x.may_have_empty_groups = true; rejected.push_back(x); }
        for(const auto& x : rejected) pass &= check(x, 0);
        policy.bf16_nn_decode_mainline_variant = -1;
        policy.bf16_nn_decode_short_variant = -1;
        pass &= check(p, 0);
    }
    // The shared short ring needs K%64; unsupported K retains the old path.
    policy.enable_bf16_nn_decode_mainline = false;
    policy.enable_bf16_nn_decode_short = true;
    policy.bf16_nn_decode_short_variant = 2;
    p.m = GroupedGemmDimensionSummary::Common(17);
    for(int k : {64, 128, 192, 512, 768, 1984})
    { p.k.extent = k; pass &= check(p, 6012); }
    for(int k : {96, 160, 2016})
    { p.k.extent = k; pass &= check(p, 0); }
    for(int variant : {3, 4, 5})
    {
        policy.bf16_nn_decode_short_variant = variant;
        for(int k : {64, 128, 192, 512, 768, 1984})
        { p.k.extent = k; pass &= check(p, 6016 + variant - 3); }
        for(int k : {96, 160, 2016})
        { p.k.extent = k; pass &= check(p, 0); }
    }
    policy.enable_bf16_nn_decode_mainline = true;
    policy.enable_bf16_nn_decode_short = false;
    for(int variant : {12, 13, 14, 15, 16, 17, 18, 19, 20})
    {
        policy.bf16_nn_decode_mainline_variant = variant;
        for(int m : {1, 16, 17, 32, 33, 64})
        for(int k : {2048, 3072, 4096, 6144, 7168, 8192})
        {
            p.m = GroupedGemmDimensionSummary::Common(m);
            p.k.extent = k;
            pass &= check(p, main_expected(variant, m, k));
        }
    }
    // New resource routes must preserve legacy fallback on disabled requests
    // and geometry outside their contract, independent of their numeric ID.
    for(int resource_variant : {17, 18, 19, 20})
    {
        policy.bf16_nn_decode_mainline_variant = resource_variant;
        p.m = GroupedGemmDimensionSummary::Common(32);
        p.n = GroupedGemmDimensionSummary::Common(1024);
        p.k = GroupedGemmDimensionSummary::Common(2048);
        policy.enable_bf16_nn_decode_mainline = false;
        pass &= check(p, 0);
        policy.enable_bf16_nn_decode_mainline = true;
        for(int resource_n : {127, 1232, 4097})
        { auto x = p; x.n.extent = resource_n; pass &= check(x, 0); }
        for(int resource_m : {0, 65, 256})
        { auto x = p; x.m.extent = resource_m; pass &= check(x, 0); }
        for(int resource_k : {1984, 2049, 8193})
        { auto x = p; x.k.extent = resource_k; pass &= check(x, 0); }
        { auto x = p; x.group_lengths_are_ragged = true; pass &= check(x, 0); }
        { auto x = p; x.may_have_empty_groups = true; pass &= check(x, 0); }
        { auto x = p; x.k = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(65536); pass &= check(x, 0); }
        { auto x = p; x.n = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(32768); pass &= check(x, 0); }
    }
    policy.bf16_nn_decode_mainline_variant = 21;
    pass &= check(p, 0);
    std::cout << "DECODE_MAINLINE_SELECTOR_" << (pass ? "PASS" : "FAIL")
              << " cases=" << checks << '\n';
    return pass ? 0 : 1;
}

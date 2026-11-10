// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <iostream>
#include <vector>

int main()
{
    using namespace ck_tile;
    GroupedGemmSelectorPolicy enabled{};
    enabled.enable_bf16_v7_nn = true;
    enabled.enable_bf16_v7_nn_small_m = true;
    enabled.enable_bf16_nn_decode_depth8 = true;
    auto disabled = enabled;
    disabled.enable_bf16_nn_decode_depth8 = false;
    GroupedGemmProblem p{};
    p.arch = GroupedGemmArch::gfx936;
    p.data_type = GroupedGemmDataType::bf16;
    p.layout = GroupedGemmLayout::nn;
    p.group_count = 64;
    p.n = GroupedGemmDimensionSummary::Common(256);
    p.k = GroupedGemmDimensionSummary::Common(512);
    int checked = 0;
    auto check = [&](const GroupedGemmProblem& x, bool expected) {
        const auto selected = select_grouped_gemm_candidate(x, enabled);
        const auto original = select_grouped_gemm_candidate(x, disabled);
        ++checked;
        if(expected ? selected.instance_id != GroupedGemmInstanceId::bf16_nn_decode_depth8
                    : selected.instance_id != original.instance_id || selected.reason != original.reason)
        {
            std::cerr << "DEPTH8_SELECTOR_FAIL case=" << checked << " m=" << x.m.extent
                      << " expected_new=" << expected << " actual=" << int(selected.instance_id)
                      << " original=" << int(original.instance_id) << '\n';
            return false;
        }
        return true;
    };
    bool pass = !GroupedGemmSelectorPolicy{}.enable_bf16_nn_decode_depth8;
    for(int m = 1; m <= 32; ++m)
    {
        p.m = GroupedGemmDimensionSummary::Common(m);
        pass &= check(p, true);
        std::vector<GroupedGemmProblem> rejected;
        for(auto arch : {GroupedGemmArch::gfx938, GroupedGemmArch::gfx946, GroupedGemmArch::unsupported})
        { auto x = p; x.arch = arch; rejected.push_back(x); }
        { auto x = p; x.data_type = GroupedGemmDataType::fp16; rejected.push_back(x); }
        for(auto layout : {GroupedGemmLayout::nt, GroupedGemmLayout::tn})
        { auto x = p; x.layout = layout; rejected.push_back(x); }
        for(int g : {0, 1, 4, 32, 63, 65, 128, 1024})
        { auto x = p; x.group_count = g; rejected.push_back(x); }
        for(int n : {0, 128, 248, 255, 264, 384, 512})
        { auto x = p; x.n.extent = n; rejected.push_back(x); }
        for(int k : {0, 32, 128, 256, 511, 640, 1024})
        { auto x = p; x.k.extent = k; rejected.push_back(x); }
        { auto x = p; x.m = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(m * 64); rejected.push_back(x); }
        { auto x = p; x.k = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(512 * 64); rejected.push_back(x); }
        { auto x = p; x.group_lengths_are_ragged = true; rejected.push_back(x); }
        { auto x = p; x.may_have_empty_groups = true; rejected.push_back(x); }
        for(const auto& x : rejected) pass &= check(x, false);
    }
    for(int m : {-1, 0, 33, 64, 128, 256, 16384})
    { p.m = GroupedGemmDimensionSummary::Common(m); pass &= check(p, false); }
    std::cout << "DEPTH8_SELECTOR_" << (pass ? "PASS" : "FAIL") << " cases=" << checked << '\n';
    return pass ? 0 : 1;
}

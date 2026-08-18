// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ck_tile;

struct TestCase
{
    const char* name;
    GroupedGemmProblem problem;
    GroupedGemmSelectorPolicy policy;
    GroupedGemmInstanceId expected;
};

GroupedGemmProblem make_problem(GroupedGemmArch arch,
                                GroupedGemmDataType data_type,
                                GroupedGemmLayout layout,
                                std::int32_t groups,
                                std::int64_t m,
                                std::int64_t n,
                                std::int64_t k)
{
    const bool variable_k = layout == GroupedGemmLayout::tn;
    return {arch,
            data_type,
            layout,
            groups,
            variable_k ? GroupedGemmDimensionSummary::Common(m)
                       : GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(m),
            GroupedGemmDimensionSummary::Common(n),
            variable_k ? GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(k)
                       : GroupedGemmDimensionSummary::Common(k),
            false,
            true};
}

std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    for(std::string field; std::getline(stream, field, ',');)
    {
        fields.push_back(field);
    }
    return fields;
}

GroupedGemmArch parse_arch(const std::string& value)
{
    if(value == "gfx936")
        return GroupedGemmArch::gfx936;
    if(value == "gfx938")
        return GroupedGemmArch::gfx938;
    return GroupedGemmArch::unsupported;
}

GroupedGemmDataType parse_data_type(const std::string& value)
{
    if(value == "fp16")
        return GroupedGemmDataType::fp16;
    if(value == "bf16")
        return GroupedGemmDataType::bf16;
    return GroupedGemmDataType::unsupported;
}

GroupedGemmLayout parse_layout(const std::string& value)
{
    if(value == "NT")
        return GroupedGemmLayout::nt;
    if(value == "NN")
        return GroupedGemmLayout::nn;
    if(value == "TN")
        return GroupedGemmLayout::tn;
    return GroupedGemmLayout::unsupported;
}

int run_table_tests()
{
    const GroupedGemmSelectorPolicy defaults{};
    GroupedGemmSelectorPolicy no_bw = defaults;
    no_bw.enable_bw_family          = false;
    GroupedGemmSelectorPolicy no_dsreadm = no_bw;
    no_dsreadm.enable_gfx936_dsreadm    = false;
    GroupedGemmSelectorPolicy no_mls = defaults;
    no_mls.enable_gfx938_mls        = false;
    GroupedGemmSelectorPolicy no_tail = defaults;
    no_tail.enable_bf16_tn_logical_k_tail = false;
    GroupedGemmSelectorPolicy no_bw_selected = defaults;
    no_bw_selected.enable_bw_selected        = false;
    GroupedGemmSelectorPolicy no_mls_large = defaults;
    no_mls_large.enable_gfx938_mls_large   = false;
    GroupedGemmSelectorPolicy family_b_mle8_enabled = defaults;
    family_b_mle8_enabled.enable_v6_family_b_m32_mle8_gfx936_candidate = true;
    GroupedGemmSelectorPolicy p1_family_a_candidate = defaults;
    p1_family_a_candidate.enable_v4_128_fp16_full_padding_candidate = true;
    GroupedGemmSelectorPolicy p1_family_a_candidate_k32 = defaults;
    p1_family_a_candidate_k32.enable_v4_128_fp16_full_padding_candidate_k32 = true;
    GroupedGemmSelectorPolicy p1_family_a_candidate_m_threshold = defaults;
    p1_family_a_candidate_m_threshold.enable_v4_fp16_m_threshold_candidate = true;
    GroupedGemmSelectorPolicy p1_family_a_candidate_128x64 = defaults;
    p1_family_a_candidate_128x64.enable_v4_128x64_fp16_full_padding_candidate = true;
    GroupedGemmSelectorPolicy p1_family_a_candidate_grid2 = defaults;
    p1_family_a_candidate_grid2.enable_v4_128_fp16_full_padding_grid2_candidate = true;
    GroupedGemmSelectorPolicy p1_family_a_candidate_v3_single_lds = defaults;
    p1_family_a_candidate_v3_single_lds.enable_v3_128_fp16_full_padding_candidate = true;
    GroupedGemmSelectorPolicy p1_family_a_candidate_v3_single_lds_gfx936 = defaults;
    p1_family_a_candidate_v3_single_lds_gfx936
        .enable_v3_128_fp16_full_padding_gfx936_candidate = true;

    auto empty_ragged = make_problem(GroupedGemmArch::gfx936,
                                     GroupedGemmDataType::fp16,
                                     GroupedGemmLayout::nt,
                                     5,
                                     5 * 1920,
                                     2048,
                                     2048);
    empty_ragged.may_have_empty_groups    = true;
    empty_ragged.group_lengths_are_ragged = true;

    auto invalid_two_variable = make_problem(GroupedGemmArch::gfx936,
                                             GroupedGemmDataType::fp16,
                                             GroupedGemmLayout::nt,
                                             4,
                                             8192,
                                             2048,
                                             2048);
    invalid_two_variable.k = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(8192);

    const std::vector<TestCase> cases{
        {"gfx936 regular threshold below",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 1919, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"gfx936 regular threshold at",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 1920, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"gfx936 TN threshold below",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::tn, 4, 2047, 2048, 8192),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"gfx936 TN logical K tail",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::tn, 4, 2048, 2048, 8191),
         defaults,
         GroupedGemmInstanceId::bw_family_logical_k_tail},
        {"gfx936 TN aligned rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::tn, 4, 2048, 2048, 8192),
         no_tail,
         GroupedGemmInstanceId::bw_family_blas_transposed},
        {"gfx936 BF16 NN first promoted family",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 16, 16 * 2048, 7168, 4096),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8},
        {"gfx936 BF16 NN second promoted family",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 16, 16 * 2048, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3},
        {"gfx936 V3 NT rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 4, 4 * 2048, 2048, 2048),
         no_bw,
         GroupedGemmInstanceId::gfx936_v3_256_m_only_padding},
        {"gfx936 BW selected capability rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 4, 4 * 2048, 2048, 2048),
         no_bw_selected,
         GroupedGemmInstanceId::gfx936_v3_256_m_only_padding},
        {"gfx936 DSReadM NN rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 4, 4 * 2048, 2048, 2048),
         no_bw,
         GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding},
        {"gfx936 default backward when DSReadM disabled",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 4, 4 * 2048, 2048, 2048),
         no_dsreadm,
         GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding},
        {"gfx938 MLS large",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 4, 4 * 2048, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_large_256},
        {"gfx938 MLS small NT",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 512, 512, 512),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_small_128},
        {"gfx938 MLS large capability rollback",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 1, 2048, 2048, 2048),
         no_mls_large,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"gfx938 MLS disabled aligned V4",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 512, 512, 512),
         no_mls,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"BF16 K tail V4 padding",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nt, 1, 512, 512, 513),
         defaults,
         GroupedGemmInstanceId::v4_128_full_padding},
        {"FP16 N aligned fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 511, 512, 511),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"full fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 511, 513, 511),
         defaults,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 Family A candidate below M boundary keeps fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 32, 1232, 1536),
         p1_family_a_candidate,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 Family A candidate at M boundary",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate,
         GroupedGemmInstanceId::v4_128_m4_full_padding_candidate},
        {"P1 candidate remains generic on gfx938",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 8, 8 * 128, 1232, 2048),
         p1_family_a_candidate,
         GroupedGemmInstanceId::v4_128_m4_full_padding_candidate},
        {"P1 candidate does not replace BF16 full padding",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 8, 8 * 128, 1232, 1536),
         p1_family_a_candidate,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 K32 candidate selects gfx938 at M boundary",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_k32,
         GroupedGemmInstanceId::v4_128_m4_k32_full_padding_candidate},
        {"P1 K32 candidate remains disabled on gfx936",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_k32,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 M-threshold candidate selects gfx938 at M boundary",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_m_threshold,
         GroupedGemmInstanceId::v4_m_threshold_full_padding_candidate},
        {"P1 M-threshold candidate remains disabled on gfx936",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_m_threshold,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 128x64 candidate selects gfx938 at M boundary",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_128x64,
         GroupedGemmInstanceId::v4_128x64_m4_full_padding_candidate},
        {"P1 128x64 candidate remains disabled on gfx936",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_128x64,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 grid2 candidate selects gfx938 at M boundary",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_grid2,
         GroupedGemmInstanceId::v4_128_m4_full_padding_grid2_candidate},
        {"P1 grid2 candidate remains disabled on gfx936",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_grid2,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 V3 single-LDS candidate selects gfx938 at M boundary",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_v3_single_lds,
         GroupedGemmInstanceId::v3_128_m4_full_padding_candidate},
        {"P1 V3 single-LDS candidate remains disabled on gfx936",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_v3_single_lds,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 V3 single-LDS gfx936 candidate selects gfx936 at M boundary",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_v3_single_lds_gfx936,
         GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936_candidate},
        {"P1 V3 single-LDS gfx936 candidate remains disabled on gfx938",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_candidate_v3_single_lds_gfx936,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"Family-B selector-effective M8 exact shape default off",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 8, 3072, 1232),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B gfx936 selector-effective M8 candidate enabled inclusive boundary",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 8, 3072, 1232),
         family_b_mle8_enabled,
         GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936_candidate},
        {"Family-B gfx936 selector-effective M9 production fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 9, 3072, 1232),
         family_b_mle8_enabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B gfx938 remains on fallback",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 8, 3072, 1232),
         family_b_mle8_enabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B BF16 near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 1, 255, 3072, 1232),
         family_b_mle8_enabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B NT near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 255, 3072, 1232),
         family_b_mle8_enabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B N near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 1, 255, 2944, 1232),
         family_b_mle8_enabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B K near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 1, 255, 3072, 1233),
         family_b_mle8_enabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"empty and ragged metadata stays selector-neutral",
         empty_ragged,
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"invalid zero groups",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 0, 2048, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::none},
        {"invalid two variable dimensions",
         invalid_two_variable,
         defaults,
         GroupedGemmInstanceId::none},
    };

    int failures = 0;
    for(const auto& test : cases)
    {
        const auto actual = select_grouped_gemm_candidate(test.problem, test.policy);
        if(actual.instance_id != test.expected)
        {
            std::cerr << "TABLE_MISMATCH name=" << test.name
                      << " expected=" << static_cast<int>(test.expected)
                      << " actual=" << static_cast<int>(actual.instance_id) << '\n';
            ++failures;
        }
    }

    std::set<int> ids;
    std::set<std::string_view> suffixes;
    for(const auto& candidate : grouped_gemm_candidate_registry)
    {
        ids.insert(static_cast<int>(candidate.id));
        suffixes.insert(candidate.canonical_suffix);
    }
    if(ids.size() != grouped_gemm_candidate_registry.size() ||
       suffixes.size() != grouped_gemm_candidate_registry.size())
    {
        std::cerr << "REGISTRY_DUPLICATE\n";
        ++failures;
    }

    if(failures == 0)
    {
        std::cout << "SELECTOR_TABLE_PASS cases=" << cases.size()
                  << " registry=" << grouped_gemm_candidate_registry.size() << '\n';
    }
    return failures;
}

int compare_golden(const std::string& golden_path, const std::string& diff_path)
{
    std::ifstream input(golden_path);
    if(!input)
    {
        throw std::runtime_error("cannot open golden CSV: " + golden_path);
    }
    std::ofstream diff;
    if(!diff_path.empty())
    {
        diff.open(diff_path);
        if(!diff)
            throw std::runtime_error("cannot open diff CSV: " + diff_path);
        diff << "arch,dtype,layout,group_num,m_arg,n_arg,k_arg,expected,actual\n";
    }

    std::string line;
    std::getline(input, line);
    const auto header = split_csv_line(line);
    const std::vector<std::string> required{"arch", "dtype", "layout", "group_num",
                                            "m_arg", "n_arg", "k_arg", "instance_id"};
    std::vector<std::size_t> columns;
    for(const auto& name : required)
    {
        const auto it = std::find(header.begin(), header.end(), name);
        if(it == header.end())
            throw std::runtime_error("missing golden column: " + name);
        columns.push_back(static_cast<std::size_t>(it - header.begin()));
    }

    int rows = 0;
    int mismatches = 0;
    while(std::getline(input, line))
    {
        if(line.empty())
            continue;
        const auto fields = split_csv_line(line);
        for(auto column : columns)
            if(column >= fields.size())
                throw std::runtime_error("short golden row");

        const auto arch = parse_arch(fields[columns[0]]);
        const auto data_type = parse_data_type(fields[columns[1]]);
        const auto layout = parse_layout(fields[columns[2]]);
        const auto groups = static_cast<std::int32_t>(std::stol(fields[columns[3]]));
        const auto m = std::stoll(fields[columns[4]]);
        const auto n = std::stoll(fields[columns[5]]);
        const auto k = std::stoll(fields[columns[6]]);
        const auto expected = fields[columns[7]];
        const auto problem = make_problem(arch, data_type, layout, groups, m, n, k);
        const auto selection = select_grouped_gemm_candidate(problem);
        const auto actual = stable_grouped_gemm_instance_id(problem, selection);
        ++rows;
        if(actual != expected)
        {
            ++mismatches;
            if(diff)
            {
                diff << fields[columns[0]] << ',' << fields[columns[1]] << ','
                     << fields[columns[2]] << ',' << groups << ',' << m << ',' << n << ',' << k
                     << ',' << expected << ',' << actual << '\n';
            }
        }
    }

    if(mismatches == 0)
    {
        std::cout << "SELECTOR_GOLDEN_PASS rows=" << rows << '\n';
    }
    else
    {
        std::cerr << "SELECTOR_GOLDEN_MISMATCH rows=" << rows
                  << " mismatches=" << mismatches << '\n';
    }
    return mismatches;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        int failures = run_table_tests();
        std::string golden_path;
        std::string diff_path;
        for(int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if(arg == "--golden" && i + 1 < argc)
                golden_path = argv[++i];
            else if(arg == "--diff" && i + 1 < argc)
                diff_path = argv[++i];
            else
                throw std::runtime_error("usage: grouped_gemm_selector_test [--golden CSV] [--diff CSV]");
        }
        if(!golden_path.empty())
            failures += compare_golden(golden_path, diff_path);
        return failures == 0 ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "SELECTOR_TEST_ERROR " << error.what() << '\n';
        return 2;
    }
}

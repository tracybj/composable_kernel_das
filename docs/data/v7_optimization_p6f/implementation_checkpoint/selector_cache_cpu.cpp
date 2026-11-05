#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cassert>
#include <cstdio>
int main()
{
    using namespace ck_tile;
    int checks=0;
    for(int m : {1,257,2048,4352})
    for(int n : {256,768,6144})
    for(int k : {64,80,3584,7168})
    for(int groups : {1,4,17,224,1024})
    for(int walk : {1,5,8})
    for(bool noslc : {false,true})
    {
        GroupedGemmProblem p{};
        p.arch=GroupedGemmArch::gfx936;p.data_type=GroupedGemmDataType::bf16;
        p.layout=GroupedGemmLayout::nn;p.group_count=groups;
        p.m=GroupedGemmDimensionSummary::Common(m);
        p.n=GroupedGemmDimensionSummary::Common(n);
        p.k=GroupedGemmDimensionSummary::Common(k);
        GroupedGemmSelectorPolicy policy{};
        policy.enable_bf16_v7_nn=true;policy.bf16_v7_nn_cache_walk=walk;
        policy.enable_bf16_v7_nn_a_noslc=noslc;
        const int expected=walk==5 ? (noslc?5012:5011) :
                           walk==8 ? (noslc?5006:5002) : (noslc?5005:5000);
        assert(int(select_grouped_gemm_candidate(p,policy).instance_id)==expected);++checks;
        // Cache factor isolation must not capture ragged/device/other layouts.
        p.group_lengths_are_ragged=true;
        const auto s=select_grouped_gemm_candidate(p,policy);
        assert(s.instance_id!=GroupedGemmInstanceId::v7_bf16_nn_cache_m01_5 &&
               s.instance_id!=GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5);++checks;
    }
    std::printf("CACHE_SELECTOR_CPU_PASS checks=%d registry=%zu\n",checks,grouped_gemm_candidate_registry.size());
}

#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cassert>
#include <cstdio>
int main()
{
    using namespace ck_tile;
    int checks=0;
    for(int m : {1,1024,1025,1280,1792,2048,2049})
    for(int n : {3840,4096,4352,6144})
    for(int k : {3056,3072,3088,3584,4096,4112})
    for(int g : {1,28,224,1024,1025})
    for(auto arch : {GroupedGemmArch::gfx936,GroupedGemmArch::gfx938})
    {
        GroupedGemmProblem p{};p.arch=arch;
        p.data_type=GroupedGemmDataType::bf16;p.layout=GroupedGemmLayout::nn;p.group_count=g;
        p.m=GroupedGemmDimensionSummary::Common(m);
        p.n=GroupedGemmDimensionSummary::Common(n);p.k=GroupedGemmDimensionSummary::Common(k);
        GroupedGemmSelectorPolicy base{};base.enable_bf16_v7_nn=true;
        auto policy=base;policy.enable_bf16_v7_nn_cache_grid=true;
        const bool eligible=m>1024 && m<=2048 && n>4096 && k>=3072 && k<=4096 &&
                            g<=1024 && arch==GroupedGemmArch::gfx936;
        const auto selected=select_grouped_gemm_candidate(p,policy).instance_id;
        if(eligible)assert(selected==GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5);
        else assert(selected==select_grouped_gemm_candidate(p,base).instance_id);
        ++checks;
        for(int invalid=0;invalid<3;++invalid)
        {
            auto q=p;
            if(invalid==0)q.group_lengths_are_ragged=true;
            if(invalid==1)q.may_have_empty_groups=true;
            if(invalid==2)q.m=GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(m*g);
            assert(select_grouped_gemm_candidate(q,policy).instance_id==select_grouped_gemm_candidate(q,base).instance_id);++checks;
        }
    }
    std::printf("CACHE_GRID_SELECTOR_CPU_PASS checks=%d\n",checks);
}

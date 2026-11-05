#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cassert>
#include <cstdio>
int main()
{
    using namespace ck_tile;
    int checks=0;
    for(int m : {1,7,63,64,65,127,128,129,256})
    for(int n : {120,128,136,248,256,264,1232,1233,1280})
    for(int k : {63,64,80,96,128,160,208,512,1536,2048,4096,4112})
    for(int g : {1,4,17,392,1024,1025})
    {
        GroupedGemmProblem p{};p.arch=GroupedGemmArch::gfx936;
        p.data_type=GroupedGemmDataType::bf16;p.layout=GroupedGemmLayout::nn;p.group_count=g;
        p.m=GroupedGemmDimensionSummary::Common(m);
        p.n=GroupedGemmDimensionSummary::Common(n);p.k=GroupedGemmDimensionSummary::Common(k);
        GroupedGemmSelectorPolicy base{};base.enable_bf16_v7_nn=true;
        auto tail=base;tail.enable_bf16_v7_nn_tail128=true;
        const bool eligible=m<=128 && n>=128 && n%8==0 && n%128!=0 &&
            k>=64 && k<=4096 && k%16==0 && g<=1024;
        auto selected=select_grouped_gemm_candidate(p,tail).instance_id;
        assert((selected==GroupedGemmInstanceId::v7_bf16_nn_tail128)==eligible);++checks;
        if(!eligible){assert(selected==select_grouped_gemm_candidate(p,base).instance_id);++checks;}
        auto wide=tail;wide.bf16_v7_nn_tail_tile_n=256;
        selected=select_grouped_gemm_candidate(p,wide).instance_id;
        assert((selected==GroupedGemmInstanceId::v7_bf16_nn_tail256)==eligible);++checks;
        if(!eligible){assert(selected==select_grouped_gemm_candidate(p,base).instance_id);++checks;}
        for(int mode : {5,6,7})
        {
            auto pol=base;pol.bf16_v7_nn_small_m_experiment=mode;
            const bool valid=m<=64 && n%128==0 && g<=1024 &&
                k>=(mode==5?128:64) && k<=4096 && k%(mode==5?32:16)==0;
            const auto id=mode==5?GroupedGemmInstanceId::v7_bf16_nn_small_m_k32_steady:
                          mode==6?GroupedGemmInstanceId::v7_bf16_nn_small_m_k16_steady:
                                  GroupedGemmInstanceId::v7_bf16_nn_small_m_noslc;
            selected=select_grouped_gemm_candidate(p,pol).instance_id;
            assert((selected==id)==valid);++checks;
            if(!valid){assert(selected==select_grouped_gemm_candidate(p,base).instance_id);++checks;}
        }
        p.group_lengths_are_ragged=true;
        assert(select_grouped_gemm_candidate(p,tail).instance_id!=GroupedGemmInstanceId::v7_bf16_nn_tail128);++checks;
        assert(select_grouped_gemm_candidate(p,wide).instance_id!=GroupedGemmInstanceId::v7_bf16_nn_tail256);++checks;
        p.group_lengths_are_ragged=false;p.may_have_empty_groups=true;
        assert(select_grouped_gemm_candidate(p,tail).instance_id!=GroupedGemmInstanceId::v7_bf16_nn_tail128);++checks;
        assert(select_grouped_gemm_candidate(p,wide).instance_id!=GroupedGemmInstanceId::v7_bf16_nn_tail256);++checks;
    }
    std::printf("TAIL_STEADY_SELECTOR_CPU_PASS checks=%d\n",checks);
}

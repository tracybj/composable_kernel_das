#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cstdlib>
#include <iostream>
int main(int argc,char** argv)
{
    if(argc!=6)return 2;
    using namespace ck_tile;
    GroupedGemmProblem p;
    p.arch=GroupedGemmArch::gfx936;
    p.data_type=GroupedGemmDataType::bf16;
    p.layout=GroupedGemmLayout::nn;
    p.m=GroupedGemmDimensionSummary::Common(std::atoll(argv[1]));
    p.n=GroupedGemmDimensionSummary::Common(std::atoll(argv[2]));
    p.k=GroupedGemmDimensionSummary::Common(std::atoll(argv[3]));
    p.group_count=std::atoi(argv[4]);
    GroupedGemmSelectorPolicy q;
    q.enable_bf16_v7_nn=true;
    switch(std::atoi(argv[5]))
    {
    case 0:break;
    case 1:q.enable_bf16_v7_nn_small_m=true;q.bf16_v7_nn_small_m_tile_n=128;break;
    case 2:q.bf16_v7_nn_small_m_experiment=1;break;
    case 3:q.bf16_v7_nn_small_m_experiment=2;break;
    default:return 2;
    }
    std::cout<<static_cast<int>(select_grouped_gemm_candidate(p,q).instance_id)<<'\n';
}

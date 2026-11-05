#include "/workspace/composable_kernel/include/ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cstdio>
int main(){int id,g,expected,v7,m01,oldn,cache,experiment;long long m,n,k;
while(std::scanf("%d %lld %lld %lld %d %d %d %d %d %d %d",&id,&m,&n,&k,&g,&expected,&v7,&m01,&oldn,&cache,&experiment)==11){
ck_tile::GroupedGemmProblem p{};p.arch=ck_tile::GroupedGemmArch::gfx936;p.data_type=ck_tile::GroupedGemmDataType::bf16;p.layout=ck_tile::GroupedGemmLayout::nn;p.group_count=g;
p.m=ck_tile::GroupedGemmDimensionSummary::Common(m);p.n=ck_tile::GroupedGemmDimensionSummary::Common(n);p.k=ck_tile::GroupedGemmDimensionSummary::Common(k);
ck_tile::GroupedGemmSelectorPolicy policy{};policy.enable_bf16_v7_nn=v7;policy.enable_bf16_v7_nn_m01_8=m01;
policy.enable_bf16_v7_nn_small_m=oldn!=0;policy.bf16_v7_nn_small_m_tile_n=oldn;policy.enable_bf16_v7_nn_a_noslc=cache;policy.bf16_v7_nn_small_m_experiment=experiment;
auto result=ck_tile::select_grouped_gemm_candidate(p,policy);std::printf("%d,%d,%d\n",id,expected,int(result.instance_id));}}

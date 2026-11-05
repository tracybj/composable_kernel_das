#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cstdio>
int main(){int id,g,expected,v7,m01,sm,cache,walk,tail,experiment,grid;long long m,n,k;
while(std::scanf("%d %lld %lld %lld %d %d %d %d %d %d %d %d %d %d",&id,&m,&n,&k,&g,&expected,&v7,&m01,&sm,&cache,&walk,&tail,&experiment,&grid)==14){
ck_tile::GroupedGemmProblem p{};p.arch=ck_tile::GroupedGemmArch::gfx936;p.data_type=ck_tile::GroupedGemmDataType::bf16;p.layout=ck_tile::GroupedGemmLayout::nn;p.group_count=g;
p.m=ck_tile::GroupedGemmDimensionSummary::Common(m);p.n=ck_tile::GroupedGemmDimensionSummary::Common(n);p.k=ck_tile::GroupedGemmDimensionSummary::Common(k);
ck_tile::GroupedGemmSelectorPolicy q{};q.enable_bf16_v7_nn=v7;q.enable_bf16_v7_nn_m01_8=m01;q.enable_bf16_v7_nn_small_m=sm!=0;q.bf16_v7_nn_small_m_tile_n=sm;
q.enable_bf16_v7_nn_a_noslc=cache;q.bf16_v7_nn_cache_walk=walk;q.enable_bf16_v7_nn_tail128=tail!=0;q.bf16_v7_nn_tail_tile_n=tail==2?256:128;q.bf16_v7_nn_small_m_experiment=experiment;q.enable_bf16_v7_nn_cache_grid=grid;
auto selected=ck_tile::select_grouped_gemm_candidate(p,q);std::printf("%d %d %d\n",id,expected,int(selected.instance_id));}}

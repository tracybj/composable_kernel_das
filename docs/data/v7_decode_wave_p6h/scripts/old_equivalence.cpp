#include HEADER
#include <iostream>
using namespace ck_tile;
int main(){
for(auto arch:{GroupedGemmArch::gfx936,GroupedGemmArch::gfx938,GroupedGemmArch::gfx946})
for(auto dtype:{GroupedGemmDataType::bf16,GroupedGemmDataType::fp16})
for(auto layout:{GroupedGemmLayout::nn,GroupedGemmLayout::nt,GroupedGemmLayout::tn})
for(int m:{1,16,32,64,128,256,512,2048})
for(int n:{128,256,1232,2048,6144})
for(int k:{32,512,768,1536,2048,4096,6144})
for(int g:{1,4,16,64,392,1024})
for(int mode=0;mode<8;++mode){
 GroupedGemmProblem p;p.arch=arch;p.data_type=dtype;p.layout=layout;p.group_count=g;
 p.m=GroupedGemmDimensionSummary::Common(m);p.n=GroupedGemmDimensionSummary::Common(n);p.k=GroupedGemmDimensionSummary::Common(k);
 GroupedGemmSelectorPolicy q;q.enable_bf16_v7_nn=true;q.enable_bf16_v7_nn_tail128=true;q.bf16_v7_nn_tail_tile_n=256;
 q.enable_bf16_v7_nn_small_m=true;q.bf16_v7_nn_small_m_experiment=mode;
 std::cout<<static_cast<int>(select_grouped_gemm_candidate(p,q).instance_id)<<'\n';
}}

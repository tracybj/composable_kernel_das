#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cassert>
#include <iostream>
using namespace ck_tile;
int main(){
 GroupedGemmSelectorPolicy off,on;on.enable_bf16_nn_decode_wave=true;
 long checks=0,active=0;
 for(auto arch:{GroupedGemmArch::gfx936,GroupedGemmArch::gfx938,GroupedGemmArch::gfx946})
 for(auto dtype:{GroupedGemmDataType::bf16,GroupedGemmDataType::fp16})
 for(auto layout:{GroupedGemmLayout::nn,GroupedGemmLayout::nt,GroupedGemmLayout::tn})
 for(int m:{1,15,16,17,32,64,65,128,256})
 for(int n:{64,127,128,256,1232,2048,6144,1048576})
 for(int k:{16,31,32,64,96,512,768,2048,4096,4128})
 for(int g:{1,4,64,1024,1025}){
  GroupedGemmProblem p;p.arch=arch;p.data_type=dtype;p.layout=layout;p.group_count=g;
  p.m=GroupedGemmDimensionSummary::Common(m);p.n=GroupedGemmDimensionSummary::Common(n);p.k=GroupedGemmDimensionSummary::Common(k);
  auto base=select_grouped_gemm_candidate(p,off);
  bool expected=arch==GroupedGemmArch::gfx936&&dtype==GroupedGemmDataType::bf16&&layout==GroupedGemmLayout::nn&&m<=64&&n%128==0&&k>=32&&k<=4096&&k%32==0&&g<=1024&&n<=2147483647LL/(2*k);
  auto got=select_grouped_gemm_candidate(p,on);bool selected=got.instance_id==GroupedGemmInstanceId::bf16_nn_decode_wave;assert(selected==expected);checks++;active+=selected;
  if(!expected)assert(got.instance_id==base.instance_id);
  if(expected){p.may_have_empty_groups=true;assert(select_grouped_gemm_candidate(p,on).instance_id!=GroupedGemmInstanceId::bf16_nn_decode_wave);p.may_have_empty_groups=false;p.group_lengths_are_ragged=true;assert(select_grouped_gemm_candidate(p,on).instance_id!=GroupedGemmInstanceId::bf16_nn_decode_wave);p.group_lengths_are_ragged=false;p.m=GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(m*g);assert(select_grouped_gemm_candidate(p,on).instance_id!=GroupedGemmInstanceId::bf16_nn_decode_wave);}
 }
 assert(static_cast<int>(GroupedGemmInstanceId::bf16_nn_decode_wave)==6000);
 assert(grouped_gemm_candidate_registry.size()==37);
 std::cout<<"DECODE_SELECTOR_PASS checks="<<checks<<" active="<<active<<"\n";
}

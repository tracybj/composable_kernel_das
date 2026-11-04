"""CPU-only sweep: prove new opt-in hooks preserve existing dataset selections."""
import csv,hashlib,json,subprocess
from pathlib import Path
R=Path('/workspace/composable_kernel');P=Path(__file__).parent
old=P/'selector_cache_reference.hpp';old.write_bytes(subprocess.check_output(['git','show','1224cca:include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'],cwd=R))
rows=list(csv.DictReader(open('/workspace/shapes_1178.csv',encoding='utf-8-sig')))
source='''#include SELECTOR_HEADER
#include <cstdio>
struct Case {int id,g;long long m,n,k;};
const Case cases[]={'''+','.join('{'+','.join(x[k] for k in ['ID','G','M','N','K'])+'}' for x in rows)+'''};
int main(){for(const auto& c:cases)for(bool v7:{false,true})for(bool cache:{false,true})for(int small:{0,128,64}){
ck_tile::GroupedGemmProblem p{};p.arch=ck_tile::GroupedGemmArch::gfx936;p.data_type=ck_tile::GroupedGemmDataType::bf16;p.layout=ck_tile::GroupedGemmLayout::nn;p.group_count=c.g;p.m=ck_tile::GroupedGemmDimensionSummary::Common(c.m);p.n=ck_tile::GroupedGemmDimensionSummary::Common(c.n);p.k=ck_tile::GroupedGemmDimensionSummary::Common(c.k);
ck_tile::GroupedGemmSelectorPolicy policy{};policy.enable_bf16_v7_nn=v7;policy.enable_bf16_v7_nn_a_noslc=cache;policy.enable_bf16_v7_nn_small_m=small!=0;policy.bf16_v7_nn_small_m_tile_n=small;
auto s=ck_tile::select_grouped_gemm_candidate(p,policy);auto name=ck_tile::stable_grouped_gemm_instance_id(p,s);std::printf("%d,%d,%d,%d,%d,%d,%s\\n",c.id,v7,cache,small,int(s.instance_id),int(s.reason),name.c_str());}}
'''
cpp=P/'selector_equivalence.cpp';cpp.write_text(source)
outputs=[]
for tag,header in [('cache_reference',old),('working',R/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp')]:
 binary=P/('selector_equivalence_'+tag)
 subprocess.run(['c++','-O2','-std=c++17','-DSELECTOR_HEADER="'+str(header)+'"',str(cpp),'-o',str(binary)],check=True)
 output=subprocess.check_output([str(binary)]);(P/(tag+'_selections.csv')).write_bytes(output);outputs.append(output)
assert outputs[0]==outputs[1]
result=dict(passed=True,cpu_only=True,shape_count=len(rows),settings_per_shape=12,total_cases=len(outputs[0].splitlines()),compared=['id','reason','stable_name'],reference_source_commit='1224cca',working_selector_sha256=hashlib.sha256((R/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp').read_bytes()).hexdigest(),output_sha256=hashlib.sha256(outputs[0]).hexdigest())
(P/'selector_equivalence.json').write_text(json.dumps(result,indent=2));print('EXISTING_SELECTOR_DATASET_EQUIVALENCE_PASS',result)

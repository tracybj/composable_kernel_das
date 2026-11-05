"""Independent source/median/count checks and CPU selector reachability of both mosaics."""
import csv,hashlib,json,math,statistics,subprocess
from pathlib import Path
import runtime as r

p=r.ROOT
base={int(x['ID']):x for x in csv.DictReader(open(r.REPO/'docs/data/v7_best_known/best_current_1193.csv'))}
shapes={int(x['ID']):x for x in csv.DictReader(open('/workspace/shapes_1178.csv',encoding='utf-8-sig'))}
raw={x['key']:x for x in [json.loads(s) for s in (p/'jobs.jsonl').read_text().splitlines()] if x.get('ok')}
names=['policy_overlay_incremental_1193','historical_envelope_incremental_1193']
checks=[];inputs=[]
def yes(e,k):return e.get(k)=='1'
for name in names:
    rows=json.loads((p/(name+'.json')).read_text());assert [x['ID'] for x in rows]==list(range(1,1194))
    for x in rows:
        i=x['ID'];s=shapes[i]
        assert all(x[k]==int(s[k]) for k in ['M','N','K','G'])
        assert x['phase']==('decode' if x['M']<=256 else 'prefill')
        assert math.isclose(x['R_frozen'],x['frozen_BLAS_ms']/x['CK_ms'],rel_tol=1e-12)
        if x['source_type']=='historical_current_retained':
            assert x['CK_ms']==float(base[i]['best_ms']) and x['instance_id']==int(base[i]['best_instance_id'])
        else:
            jobs=[raw[k] for k in x['source_keys']]
            assert len(jobs)==3 and {t['round'] for t in jobs}=={1,2,3}
            assert {t['selected'] for t in jobs}=={x['instance_id']}
            assert len({t['epoch'] for t in jobs})==len({t['binary_sha256'] for t in jobs})==1
            assert statistics.median(t['result']['median_ms'] for t in jobs)==x['CK_ms']
            assert all(t['kind']=='perf' and t['physical_gpu']==4 for t in jobs)
            assert x['full_output_accuracy_keys']
        if name.startswith('historical_envelope'):assert x['CK_ms']<=float(base[i]['best_ms'])
        e=x['enable_env'];enabled=yes(e,'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN') and not yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN')
        m01=not yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8')
        sm=int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M','0'))
        sm=sm if sm in [64,128] and not yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M') else 0
        noslc=yes(e,'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC') and not yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_A_NOSLC')
        walk=int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK','0'))
        walk=walk if walk in [1,5,8] and not yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_WALK') else 0
        tail=int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL','0'))
        tail=tail if tail in [1,2] and not yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_N_TAIL') else 0
        exp=int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT','0'))
        if yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M') or yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M_EXPERIMENT'):exp=0
        grid=yes(e,'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_GRID') and not yes(e,'CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_GRID')
        inputs.append(' '.join(map(str,[i,x['M'],x['N'],x['K'],x['G'],x['instance_id'],int(enabled),int(m01),sm,int(noslc),walk,tail,exp,int(grid)])))
        checks.append(dict(view=name,ID=i,expected=x['instance_id']))

cpp=p/'selector_incremental_probe.cpp'
cpp.write_text('''#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cstdio>
int main(){int id,g,expected,v7,m01,sm,cache,walk,tail,experiment,grid;long long m,n,k;
while(std::scanf("%d %lld %lld %lld %d %d %d %d %d %d %d %d %d %d",&id,&m,&n,&k,&g,&expected,&v7,&m01,&sm,&cache,&walk,&tail,&experiment,&grid)==14){
ck_tile::GroupedGemmProblem p{};p.arch=ck_tile::GroupedGemmArch::gfx936;p.data_type=ck_tile::GroupedGemmDataType::bf16;p.layout=ck_tile::GroupedGemmLayout::nn;p.group_count=g;
p.m=ck_tile::GroupedGemmDimensionSummary::Common(m);p.n=ck_tile::GroupedGemmDimensionSummary::Common(n);p.k=ck_tile::GroupedGemmDimensionSummary::Common(k);
ck_tile::GroupedGemmSelectorPolicy q{};q.enable_bf16_v7_nn=v7;q.enable_bf16_v7_nn_m01_8=m01;q.enable_bf16_v7_nn_small_m=sm!=0;q.bf16_v7_nn_small_m_tile_n=sm;
q.enable_bf16_v7_nn_a_noslc=cache;q.bf16_v7_nn_cache_walk=walk;q.enable_bf16_v7_nn_tail128=tail!=0;q.bf16_v7_nn_tail_tile_n=tail==2?256:128;q.bf16_v7_nn_small_m_experiment=experiment;q.enable_bf16_v7_nn_cache_grid=grid;
auto selected=ck_tile::select_grouped_gemm_candidate(p,q);std::printf("%d %d %d\\n",id,expected,int(selected.instance_id));}}
''')
cmd=['g++','-O2','-std=c++17','-I'+str(r.REPO/'include'),str(cpp),'-o',str(p/'selector_incremental_probe')]
subprocess.run(cmd,check=True)
output=subprocess.check_output([str(p/'selector_incremental_probe')],input='\n'.join(inputs)+'\n',text=True).splitlines()
assert len(output)==len(checks)==2386
for check,line in zip(checks,output):
    i,expected,actual=map(int,line.split());assert i==check['ID'] and expected==check['expected'];check['actual']=actual;assert actual==expected,check
with (p/'selector_incremental_checks.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(checks[0]),lineterminator='\n');w.writeheader();w.writerows(checks)
(p/'incremental_view_audit.json').write_text(json.dumps(dict(at=r.stamp(),passed=True,cpu_only=True,selector_checks=2386,shapes=1193,views=names,compile_command=cmd,header_sha256=r.sha(r.REPO/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'),note='Reachability and provenance only; not GPU full replay or proof of stable optimality'),indent=2))
print('INCREMENTAL_VIEW_AUDIT_PASS shapes1193 CPU_selector2386')

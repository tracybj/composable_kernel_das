#!/usr/bin/env python3
"""CPU-only availability check for every current scheme/shape recipe in the ledger."""
import csv,gzip,hashlib,json,subprocess
from pathlib import Path

P=Path(__file__).resolve().parent
R=Path('/workspace/composable_kernel')
HEADER=R/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'

def main():
    with gzip.open(P/'all_candidate_series.jsonl.gz','rt') as source:rows=[json.loads(line) for line in source]
    shapes={int(row['ID']):row for row in csv.DictReader(open('/workspace/shapes_1178.csv',encoding='utf-8-sig'))}
    choices={}
    for row in rows:
        if row['current_available'] and row['gpu']==4 and row['protocol'].startswith('card4_device_args_single'):
            choices[(row['ID'],row['scheme'])]=row
    choices=[choices[key] for key in sorted(choices)]
    cpp=P/'selector_probe.cpp'
    cpp.write_text('''#include "/workspace/composable_kernel/include/ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cstdio>
int main(){int id,g,expected,v7,m01,oldn,cache,experiment;long long m,n,k;
while(std::scanf("%d %lld %lld %lld %d %d %d %d %d %d %d",&id,&m,&n,&k,&g,&expected,&v7,&m01,&oldn,&cache,&experiment)==11){
ck_tile::GroupedGemmProblem p{};p.arch=ck_tile::GroupedGemmArch::gfx936;p.data_type=ck_tile::GroupedGemmDataType::bf16;p.layout=ck_tile::GroupedGemmLayout::nn;p.group_count=g;
p.m=ck_tile::GroupedGemmDimensionSummary::Common(m);p.n=ck_tile::GroupedGemmDimensionSummary::Common(n);p.k=ck_tile::GroupedGemmDimensionSummary::Common(k);
ck_tile::GroupedGemmSelectorPolicy policy{};policy.enable_bf16_v7_nn=v7;policy.enable_bf16_v7_nn_m01_8=m01;
policy.enable_bf16_v7_nn_small_m=oldn!=0;policy.bf16_v7_nn_small_m_tile_n=oldn;policy.enable_bf16_v7_nn_a_noslc=cache;policy.bf16_v7_nn_small_m_experiment=experiment;
auto result=ck_tile::select_grouped_gemm_candidate(p,policy);std::printf("%d,%d,%d\\n",id,expected,int(result.instance_id));}}
''')
    command=['c++','-O2','-std=c++17',str(cpp),'-o',str(P/'selector_probe')]
    subprocess.run(command,check=True,capture_output=True)
    lines=[]
    for row in choices:
        s=shapes[row['ID']];e=row['enable_env']
        values=[row['ID'],*[s[key] for key in ['M','N','K','G']],row['current_instance_id'],
            int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN','0')),
            1-int(e.get('CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8','0')),
            int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M','0')),
            int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC','0')),
            int(e.get('CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT','0'))]
        lines.append(' '.join(map(str,values)))
    output=subprocess.check_output([str(P/'selector_probe')],input=('\n'.join(lines)+'\n').encode()).decode().splitlines()
    assert len(output)==len(choices)
    checks=[]
    for row,line in zip(choices,output):
        i,expected,actual=map(int,line.split(','))
        checks.append(dict(ID=i,scheme=row['scheme'],expected=expected,actual=actual,passed=expected==actual))
    with (P/'selector_checks.csv').open('w',newline='') as target:
        writer=csv.DictWriter(target,list(checks[0]),lineterminator='\n');writer.writeheader();writer.writerows(checks)
    result=dict(passed=all(row['passed'] for row in checks),cpu_only=True,checks=len(checks),shape_count=len({row['ID'] for row in checks}),
        selector_sha256=hashlib.sha256(HEADER.read_bytes()).hexdigest(),compile_command=command,
        note='Policy-level CPU selection; does not launch GPU or newly verify performance/precision. Environment recipes assume unrelated experiment switches unset.')
    (P/'selector_audit.json').write_text(json.dumps(result,indent=2))
    assert result['passed'],[row for row in checks if not row['passed']]
    print('CURRENT_RECIPE_SELECTOR_PASS',result['checks'],'pairs',result['shape_count'],'shapes; CPU only')

if __name__=='__main__':main()

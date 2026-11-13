import sys,os,resource
sys.path.insert(0,'/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z')
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.nice(10)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
"""CPU-only equivalence of the exact old/new API policy readers, no HIP link."""
from pathlib import Path
import copy,json,random,statistics,subprocess,time
import plan,runtime as r
ROOT=Path('/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z')
REPO=Path('/workspace/composable_kernel')

def main():
 folder=Path('/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/cpu_phase_policy');folder.mkdir(exist_ok=False)
 relative='example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp'
 old=(ROOT/'compile/build9/previous/source'/relative).read_text()
 new=(REPO/relative).read_text()
 header=(REPO/'include/ck_tile/ops/gemm/grouped_gemm_bw_family_selected.hpp').read_text()
 helpers=header[header.index('inline bool ck_tile_'):header.index('// The v7 device-args entries.')]
 def body(text,start):
  return text[text.index(start,text.index('int convert_problem(')):text.index('\n}\n\nint write_selection(')]
 bodies=[body(old,'    // Only explicit new decode requests'),body(new,'    // Only explicit new decode requests')]
 code='''#include <cstdlib>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_DECODE_BF16 1
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED 1
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_V7_V8_BF16 1
'''+helpers
 for i,b in enumerate(bodies):
  code+=f'__attribute__((noinline)) int read_policy_{i}(const ck_tile::GroupedGemmProblem& problem, ck_tile::GroupedGemmSelectorPolicy& policy, unsigned disabled_flags) {{\nck_tile_hcu_grouped_gemm_problem_v1 record{{}}; record.policy_disable_flags=disabled_flags; const auto* source=&record;\n'+b+'\n}\n'
 code+='''int main() {
 std::string line; std::vector<std::string> prior;
 while(std::getline(std::cin,line)) {
  for(const auto& key:prior) unsetenv(key.c_str()); prior.clear();
  std::istringstream input(line); int id,arch,type,layout,groups,kind,empty,ragged,mode;
  long long m,n,k; unsigned disabled; size_t count;
  input>>id>>arch>>type>>layout>>m>>n>>k>>groups>>kind>>empty>>ragged>>disabled>>mode>>count;
  if(!input) return 2;
  for(size_t i=0;i<count;++i) {std::string key,value;input>>key>>value; if(value=="~")value=""; setenv(key.c_str(),value.c_str(),1);prior.push_back(key);}
  ck_tile::GroupedGemmProblem p{};p.arch=static_cast<ck_tile::GroupedGemmArch>(arch);
  p.data_type=static_cast<ck_tile::GroupedGemmDataType>(type);p.layout=static_cast<ck_tile::GroupedGemmLayout>(layout);
  p.group_count=groups;p.m=ck_tile::GroupedGemmDimensionSummary::Common(m);p.n=ck_tile::GroupedGemmDimensionSummary::Common(n);p.k=ck_tile::GroupedGemmDimensionSummary::Common(k);
  p.m.kind=static_cast<ck_tile::GroupedGemmDimensionKind>(kind);p.may_have_empty_groups=empty;p.group_lengths_are_ragged=ragged;
  ck_tile::GroupedGemmSelectorPolicy a{},b{};
  const int ra=read_policy_0(p,a,disabled),rb=read_policy_1(p,b,disabled);
  const auto sa=ck_tile::select_grouped_gemm_candidate(p,a),sb=ck_tile::select_grouped_gemm_candidate(p,b);
  std::cout<<id<<" "<<ra<<" "<<int(sa.instance_id)<<" "<<int(sa.reason)<<" "<<rb<<" "<<int(sb.instance_id)<<" "<<int(sb.reason)<<"\\n";
  if(mode) {
   using Clock=std::chrono::steady_clock;
   for(int block=0;block<11;++block)for(int side=0;side<2;++side) {
    const int arm=(side+block)%2;auto start=Clock::now();
    for(int j=0;j<20000;++j) {
     ck_tile::GroupedGemmSelectorPolicy q{};int status=arm?read_policy_1(p,q,disabled):read_policy_0(p,q,disabled);
     asm volatile("" : : "g"(&q),"g"(status) : "memory");
    }
    auto stop=Clock::now();std::cout<<"T "<<id<<" "<<block<<" "<<arm<<" "<<std::chrono::duration<double,std::nano>(stop-start).count()/20000<<"\\n";
   }
  }
 }
}
'''
 (folder/'policy_equivalence.cpp').write_text(code)
 command=['g++','-O3','-std=c++17','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-I'+str(REPO/'include'),str(folder/'policy_equivalence.cpp'),'-o',str(folder/'policy_equivalence')]
 compiled=subprocess.run(command,text=True,capture_output=True)
 (folder/'build.log').write_text(compiled.stdout+compiled.stderr);assert compiled.returncode==0,compiled.stderr
 linked=subprocess.run(['ldd',str(folder/'policy_equivalence')],text=True,capture_output=True,check=True).stdout
 (folder/'ldd.txt').write_text(linked);assert not any(x in linked.lower() for x in ['libamdhip','libgalaxyhip','libhsa','libhipblas'])
 cases=[];source=r.read(ROOT/'source_results.json');P=plan.PREFIX
 def add(shape,env,comparison=None,expected=None,properties=None,timing=False):
  item=dict(index=len(cases),shape=shape,env=env,comparison=comparison,expected=expected,properties=properties or {},timing=timing);cases.append(item);return item['index']
 for row in source:add({z:row[z] for z in ['M','N','K','G']},row['env'],'old_equals_new',row['actual_instance'])
 # Exercise every original with every new main variant and strict fast/disable values.
 for row in source:
  shape={z:row[z] for z in ['M','N','K','G']}
  for family,variants in [('MAINLINE',[6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24]),('SHORT',[3,5])]:
   for variant in variants:
    env=dict(row['env']);env[P+'ENABLE_BF16_NN_DECODE_'+family]='1';env[P+'BF16_NN_DECODE_'+family+'_VARIANT']=str(variant)
    reference=None
    for enable,disable in [('0','0'),('1','0'),('1','1'),('yes','0')]:
     e=dict(env);e[P+'ENABLE_BF16_NN_DECODE_FAST_POLICY']=enable;e[P+'DISABLE_BF16_NN_DECODE_FAST_POLICY']=disable
     idx=add(shape,e,('same_as',reference) if reference is not None else ('old_equals_new' if variant<=20 else None))
     if reference is None:reference=idx
 # Deterministic boundary/unsupported/public-disable cases including malformed switches.
 rng=random.Random(6411932)
 for i in range(1200):
  shape=dict(M=rng.choice([-1,0,1,16,17,32,33,64,65,256]),N=rng.choice([128,256,1024,1152,1232,4096,4224]),K=rng.choice([0,64,512,2048,2176,4096,6144,7168,8192,8320]),G=rng.choice([0,1,3,32,33,1024,1025]))
  env={plan.V7:'1',P+'ENABLE_BF16_NN_DECODE_MAINLINE':rng.choice(['0','1','yes','']),P+'DISABLE_BF16_NN_DECODE_MAINLINE':rng.choice(['0','1','no']),P+'BF16_NN_DECODE_MAINLINE_VARIANT':rng.choice(['0','6','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','01','-1','1x','']),P+'ENABLE_BF16_NN_DECODE_SHORT':rng.choice(['0','1']),P+'BF16_NN_DECODE_SHORT_VARIANT':rng.choice(['0','3','5','6','01']),P+'DISABLE_BF16_NN_DECODE_SHORT':rng.choice(['0','1'])}
  props=dict(arch=rng.choice([0,1,2,3,4]),type=rng.choice([0,1,2]),layout=rng.choice([0,1,2,3]),kind=rng.choice([0,1]),empty=rng.choice([0,1]),ragged=rng.choice([0,1]),disabled=rng.choice([0,1,2,4,8,16,32,63]))
  env[P+'ENABLE_BF16_NN_DECODE_FAST_POLICY']='0';ref=add(shape,env,None,properties=props)
  fast=dict(env);fast[P+'ENABLE_BF16_NN_DECODE_FAST_POLICY']='1';add(shape,fast,('same_as',ref),properties=props)
 for ident in [621,1072,76,257]:
  row=next(x for x in source if x['ID']==ident);family='MAINLINE' if ident in [621,1072] else 'SHORT'
  e=dict(row['env']);e[P+'ENABLE_BF16_NN_DECODE_'+family]='1';e[P+'BF16_NN_DECODE_'+family+'_VARIANT']='11' if family=='MAINLINE' else '3';e[P+'ENABLE_BF16_NN_DECODE_FAST_POLICY']='1'
  add({z:row[z] for z in ['M','N','K','G']},e,'old_equals_new',timing=False)
 r.write(folder/'cases.json',cases)
 lines=[]
 for c in cases:
  p=c['properties'];s=c['shape'];env=c['env'];prefix=[c['index'],p.get('arch',1),p.get('type',2),p.get('layout',2),s['M'],s['N'],s['K'],s['G'],p.get('kind',0),p.get('empty',0),p.get('ragged',0),p.get('disabled',0),int(c['timing']),len(env)]
  lines.append(' '.join(map(str,prefix))+''.join(' '+k+' '+(v or '~') for k,v in sorted(env.items())))
 start=time.monotonic();result=subprocess.run([str(folder/'policy_equivalence')],input='\n'.join(lines)+'\n',text=True,capture_output=True,env=plan.environment({}),timeout=240)
 (folder/'run.log').write_text(result.stdout+result.stderr);assert result.returncode==0
 outputs={};times={}
 for line in result.stdout.splitlines():
  x=line.split()
  if x[0]=='T':times.setdefault(int(x[1]),{0:[],1:[]})[int(x[3])].append(float(x[4]));continue
  values=list(map(int,x));outputs[values[0]]=values[1:]
 assert len(outputs)==len(cases)
 checked=0
 for c in cases:
  oldout,newout=outputs[c['index']][:3],outputs[c['index']][3:]
  if c['comparison']=='old_equals_new':assert oldout==newout,(c,oldout,newout);checked+=1
  elif c['comparison']:
   assert newout==outputs[c['comparison'][1]][3:],(c,newout,outputs[c['comparison'][1]]);checked+=1
  if c['expected'] is not None:assert newout[1]==c['expected'],(c,newout)
 timing=[dict(case=i,old_ns=statistics.median(v[0]),new_fast_ns=statistics.median(v[1]),blocks=v) for i,v in times.items()]
 report=dict(ok=True,cpu_only=True,gpu=False,case_count=len(cases),comparisons=checked,seconds=time.monotonic()-start,command=command,ldd=linked,old_source_sha256=plan.sha(ROOT/'compile/build9/previous/source'/relative),new_source_sha256=plan.sha(REPO/relative),timings=timing,note='Exact extracted policy readers and final pure selection only; excludes ABI/header validation, arch query, driver and GPU. CPU timing is not proof of formal GPU benefit.')
 r.write(folder/'results.json',report);print('H3_FAST_POLICY_CPU_EQUIVALENCE_PASS',len(cases),'comparisons',checked,flush=True)
 for x in timing:print('CPU_POLICY_TIMING_NS',x['case'],x['old_ns'],x['new_fast_ns'],flush=True)

if __name__=='__main__':main()

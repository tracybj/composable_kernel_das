#!/usr/bin/env python3
"""Aggregate every diagnostic round; preserve raw dispatches and timing spans."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import struct
import sys
from elf_isa import sections,metadata

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/v7_coverage/p4a')
from perfdump import dispatches,zblocks
spec=importlib.util.spec_from_file_location('p6metrics','/workspace/v7_coverage/p6/analyze.py')
p6=importlib.util.module_from_spec(spec);spec.loader.exec_module(p6)

def dispatch_info(path):
 # xprof HYGPERF type-9 dispatch record. Layout is locally inferred, not an
 # official interchange format. Cross-check against SQTT results.txt below.
 data=Path(path).read_bytes();zs=zblocks(data);pos=zs[0][0]+zs[0][1];end=zs[-1][0];out=[]
 while pos+32<=end:
  kind,offset,size,_=struct.unpack_from('<4Q',data,pos)
  if not size or size>end-pos:pos+=8;continue
  if kind==9:
   body=data[pos+32:pos+size];assert len(body)>=80
   dims=struct.unpack_from('<10I',body,40)
   gx,gy,gz,bx,by,bz,lds,scratch,vgpr,sgpr=dims
   assert bx*by*bz and gx*gy*gz%(bx*by*bz)==0
   out.append(dict(grid_work_items=[gx,gy,gz],block_threads=[bx,by,bz],ctas=gx*gy*gz//(bx*by*bz),lds=lds,scratch=scratch,vgpr=vgpr,sgpr=sgpr))
  pos+=size
 return out

def main():
 cases=json.loads((ROOT/'cases.json').read_text());case_byid={c['id']:c for c in cases}
 jobs=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
 good={r['key']:r for r in jobs if r.get('ok')}
 timing=[];profiles=[];raw=[]
 for c in cases:
  row={k:c[k] for k in ['id','M','N','K','G','phase','frozen_blas_ms','blas_reference_gpu']}
  for backend in ['base','small128','blas']:
   rs=[good[f'timing_r{rnd}_id{c["id"]}_{backend}'] for rnd in [1,2,3] if f'timing_r{rnd}_id{c["id"]}_{backend}' in good]
   if not rs:continue
   keys=[1,20] if backend=='blas' else [1]
   for repeat in keys:
    values=[next(t for t in r['result']['timings'] if t['repeat']==repeat) for r in rs] if backend=='blas' else [r['result'] for r in rs]
    row[backend+('_repeat20' if repeat==20 else '')]=dict(rounds=len(rs),median_ms=statistics.median(v['median_ms'] for v in values),round_median_ms=[v['median_ms'] for v in values],round_cv=[v['cv'] for v in values],range_ms=[min(v['min_ms'] for v in values),max(v['max_ms'] for v in values)])
  if 'small128' in row and 'base' in row:row['small_delta_pct']=100*(row['small128']['median_ms']/row['base']['median_ms']-1)
  if 'blas' in row:
   for backend in ['base','small128']:
    if backend in row:
     row['R_single_'+backend]=row['blas']['median_ms']/row[backend]['median_ms']
     row['R_mixed_'+backend]=row['blas_repeat20']['median_ms']/row[backend]['median_ms']
  timing.append(row)
 for r in good.values():
  if r['mode']!='profile':continue
  c=case_byid[r['id']];i=r['id'];backend=r['backend'];shape=tuple(c[k] for k in ['M','N','K','G'])
  tile=(64,128,16) if backend=='small128' else (64,128,64) if i==146 and backend=='base' else (256,256,16)
  if backend=='blas':
   native=tuple(map(int,re.search(r'MT(\d+)x(\d+)x(\d+)',c['kernel_name']).groups()));tile=(native[1],native[0],native[2])
  assert len(r['perf_files'])==1,r
  ds=dispatches(r['perf_files'][0]);assert len(ds)==2,(r['key'],len(ds))
  metric=p6.metrics(ds[-1],shape,tile)
  geometries=dispatch_info(r['perf_files'][0]);assert len(geometries)==2,(r['key'],geometries)
  for d in ds:
   assert d['SQ_INSTS_MMOP']==metric['mmop_insts_expected_padded'],(r['key'],d['SQ_INSTS_MMOP'],metric['mmop_insts_expected_padded'])
  persistent=i==146 and backend=='base'
  if persistent:
   # 4003 launches 80 persistent CTAs, then iterates 3920 logical tiles.
   # SQ_WAVES counts hardware waves, not one fresh wave per logical tile.
   assert all(d['SQ_WAVES']==320 for d in ds),(r['key'],metric['waves'])
   metric['logical_tiles']=metric.pop('ctas')
   metric.pop('waves_per_cta')
   metric['physical_ctas']=80
   metric['waves_per_physical_cta']=4
   metric['logical_tiles_per_physical_cta']=metric['logical_tiles']/80
   assert all(g['ctas']==80 for g in geometries)
  else:
   assert metric['waves_per_cta'] in [4,8],(r['key'],metric['waves_per_cta'])
   assert all(g['ctas']==metric['ctas'] for g in geometries),(r['key'],geometries,metric['ctas'])
  log=(Path(r['path'])/'run.log').read_text()
  aql=[int(x)/1e6 for x in re.findall(r'dispatch kernel .*?execution\((\d+) ns\)',log)]
  profiles.append(dict(id=i,backend=backend,tile=tile,source=r['perf_files'][0],dispatch_count=len(ds),dispatch_geometries=geometries,aql_dispatch_ms=aql,**metric))
  raw.append(dict(key=r['key'],source=r['perf_files'][0],dispatches=ds))
  if persistent:
   code=zblocks(Path(r['perf_files'][0]).read_bytes())[0][2]
   md=metadata(code,sections(code))
   matched=[(name,m) for name,m in md.items() if m.get('.private_segment_fixed_size')==132 and 'PU3AS4Kv' in name]
   assert len(matched)==1,len(matched)
   name,m=matched[0]
   assert m['.vgpr_count']==256 and m['.group_segment_fixed_size']==49152
   assert m['.vgpr_spill_count']==39 and m['.sgpr_spill_count']==106
   (ROOT/'v4_fallback_metadata.json').write_text(json.dumps(dict(source=r['perf_files'][0],code_object_sha256=hashlib.sha256(code).hexdigest(),symbol=name,metadata={k:v for k,v in m.items() if k not in ['.args','.unfolded_args']},dispatch_resources=geometries[0]),indent=2))
 for name,value in [('timing.json',timing),('profiles.json',profiles),('raw_counters.json',raw)]:
  (ROOT/name).write_text(json.dumps(value,indent=2))
 validation=dict(jobs=len(jobs),successful=len(good),failures=[r for r in jobs if not r.get('ok')],timing_jobs=sum(r['mode']=='timing' for r in good.values()),profile_jobs=len(profiles),trace_jobs=sum(r['mode']=='trace' for r in good.values()),cpu_full_blas_checks=sum(r['result'].get('checked',0) for r in good.values() if r['mode']=='timing' and r['backend']=='blas'),all_profile_mmop_matched=True,all_isa_decoded=all(r['unknown_lines']==0 for r in json.load(open(ROOT/'isa_summary.json'))))
 for r in good.values():
  if r['mode']!='trace':continue
  text=(Path(r['path'])/'profile/results.txt').read_text();parsed=dispatch_info(r['perf_files'][0]);assert len(parsed)==1
  g=parsed[0];match=re.search(r'grd\((\d+), (\d+), (\d+)\), wgr\((\d+), (\d+), (\d+)\), lds\((\d+)\), scr\((\d+)\), vgpr\((\d+)\), sgpr\((\d+)\)',text)
  assert match
  assert list(map(int,match.groups()))==g['grid_work_items']+g['block_threads']+[g[k] for k in ['lds','scratch','vgpr','sgpr']],(r['key'],g)
 validation['dispatch_record_layout_crosschecked_with_sqtt_text']=True
 (ROOT/'validation.json').write_text(json.dumps(validation,indent=2))
 print('DIAGNOSTIC_PROGRESS',validation)
 for r in timing:
  print('TIME',r['id'],*[f'{b}={r[b]["median_ms"]:.6f}({r[b]["rounds"]}r)' for b in ['base','small128','blas','blas_repeat20'] if b in r])
 for r in profiles:print('PROFILE',r['id'],r['backend'],'ms',round(r['ms_from_gui'],5),'wave_cycles',round(r['mean_wave_lifetime_cycles'],1),'wait',round(r['wait_any_frac'],3),'HBM_MB',round(r['hbm_read_MB'],2),'vmem/wave',r['sq_insts_vmem_per_wave'])

if __name__=='__main__':main()

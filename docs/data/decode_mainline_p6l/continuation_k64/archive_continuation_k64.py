"""Audit and locally archive the completed, explicitly authorized continuation."""
from pathlib import Path
import collections,datetime,gzip,hashlib,io,json,os,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
DEST=REPO/'docs/data/decode_mainline_p6l/continuation_k64'
def read(p):return json.loads(p.read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def git(*a,env=None):return subprocess.check_output(['git',*a],cwd=REPO,env=env)
def main():
 import runtime as r
 session=read(ROOT/'session.json');assert time.time()<session['gpu_deadline_epoch']
 assert read(ROOT/'k64_controller_status.json')['state']=='complete'
 assert read(ROOT/'k64_qualification_audit.json')['ok']
 analysis=read(ROOT/'k64_discovery_analysis.json');assert analysis['complete_phase'] and analysis['completed_pairs']==46
 captures=read(ROOT/'k64_profiles.json')+read(ROOT/'drafts/k64_wide_control_profile/captures.json');assert len(captures)==10 and all(x['raw_token_bytes']>=4096 and x['active_CU_index']==0 for x in captures)
 jobs=[json.loads(x) for x in (ROOT/'jobs.jsonl').read_text().splitlines()]
 started=[json.loads(x) for x in (ROOT/'started.jsonl').read_text().splitlines()]
 assert len(jobs)==len(started)==812 and all(x['ok'] and x['rc']==0 and x['physical_gpu']==4 and x['epoch']==0 for x in jobs)
 jobs=[x for x in jobs if x['benchmark_sha256']=='d94d9808dede6ae6ba6e8cd332d9e6b91196795e08a96713b13278ccc6a59374'];assert len(jobs)==407
 gates={x['key']:x for x in map(json.loads,(ROOT/'gates.jsonl').read_text().splitlines())}
 kernel={x['key']:x for x in map(json.loads,(ROOT/'kernel_checks.jsonl').read_text().splitlines())}
 comps=[];raw={};versions={};manifests={}
 for x in jobs:
  assert datetime.datetime.fromisoformat(x['at']).timestamp()<session['gpu_deadline_epoch']
  assert x['environment']['HIP_VISIBLE_DEVICES']==x['environment']['CUDA_VISIBLE_DEVICES']=='4'
  gate=gates[x['key']];assert gate['rc']==0 and gate['hcu']=='0.0%' and 'MKFD' not in gate['output'].upper() and gate['mem_available_kib']>=64*1024*1024
  assert all(kernel[x['key']+s]['rc']==0 and not kernel[x['key']+s]['events'] for s in ['_pre','_post'])
  text=(Path(x['path'])/'run.log').read_text();parsed=r.parse_output(text,x['kind'],x.get('expected'),x.get('repeats',1),x.get('regression_kind'))
  if x['kind']=='accuracy':
   assert [list(c) for c in parsed['comparisons']]==x['comparisons']
   assert all(int(c[1])==x['shape']['M']*x['shape']['N']*x['shape']['G'] and int(c[2])==0 for c in parsed['comparisons']);comps+=parsed['comparisons']
  if x['kind']=='perf':assert parsed['result']==x['result']
  for p in Path(x['path']).iterdir():
   if p.is_file():assert p.suffix not in ['.perf','.o','.co','.isa'];raw['jobs/'+p.parent.name+'/'+p.name]=p.read_bytes()
  for n,s in x['script_sha256'].items():versions[n,s]=ROOT/'source_versions'/(s+'_'+n)
  manifests[x['plan_sha256']]=ROOT/'manifests'/(x['plan_sha256']+'.json')
 assert len(comps)==357
 for (n,s),p in versions.items():data=p.read_bytes();assert sha(data)==s;raw['source_versions/'+p.name]=data
 for s,p in manifests.items():raw['manifests/'+p.name]=p.read_bytes()
 selected_keys={x['key'] for x in jobs}
 for n in ['gates.jsonl','started.jsonl','kernel_checks.jsonl']:
  accepted=[]
  for line in (ROOT/n).read_bytes().splitlines(keepends=True):
   item=json.loads(line);key=item['record']['key'] if n=='started.jsonl' else item['key']
   valid=key in selected_keys or (n=='kernel_checks.jsonl' and any(key==z+suffix for z in selected_keys for suffix in ['_pre','_post']))
   if valid:accepted.append(line)
  raw['journals/'+n]=b''.join(accepted)
 raw['journals/k64_controller_events.jsonl']=(ROOT/'k64_controller_events.jsonl').read_bytes()
 for manifest in manifests.values():
  for name,digest in read(manifest)['runner_sha256'].items():
   if name.startswith('drafts/'):
    source=ROOT/name;data=source.read_bytes();assert sha(data)==digest;raw['auxiliary_sources/'+name]=data
 deliveries=[read(ROOT/'profile_delivery_batch10.json'),read(ROOT/'profile_delivery_batch10b.json')]
 for delivery in deliveries:
  package=Path(delivery['path']);assert package.stat().st_size==delivery['bytes'] and sha(package.read_bytes())==delivery['sha256']
 assert not DEST.exists();DEST.mkdir()
 names=['session.json','authorization.json','initial_files_sha256.json','historical_exclusion_inventory.json','historical_final_attempts.json','research_plan.json','heldout_protocol.json','k64_hypothesis.json','k64_discovery_plan.json','k64_discovery_results.json','k64_discovery_analysis.json','k64_discovery_analysis.csv','k64_qualification.json','k64_qualification_audit.json','k64_profile_rule.json','k64_profile_plan.json','k64_profile_attempts.json','k64_profiles.json','profile_delivery_batch10.json','profile_delivery_batch10b.json','k64_campaign_cpu_selector_audit.json','k64_baseline1193_cpu_audit.json','k64_controller_status.json','k64_build_isa_audit.json','k64_source_commit.json','compile/build10/source_sha256.json','compile/build10/completed.json','environment_epoch.json','recovery_total.json','k64_profile_analysis/profile_analysis.json','k64_profile_analysis/h3_static_schedule.json','drafts/k64_wide_control_profile/registered_profile_plan.json','drafts/k64_wide_control_profile/captures.json','drafts/k64_wide_control_profile/attempts.json','drafts/k64_wide_control_profile/analysis/profile_analysis.json','drafts/k64_wide_control_profile/analysis/h3_static_schedule.json','drafts/k64_own_commit/failed_initial_base/reason.json','drafts/k64_own_commit/timing_quality_observation.json','cpu_k64_policy/results.json','cpu_k64_policy/public_disable_equivalence.json','cpu_k64_regressions/results.json','archive_continuation_k64.py']
 for n in names:
  p=DEST/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 for delivery in deliveries:
  package=Path(delivery['path'])
  with tarfile.open(package) as archive:
   for member in archive.getmembers():
    if Path(member.name).name in ['XCOMPUTE_README.md','profile_manifest.json','profile_manifest.csv']:
     p=DEST/'profile_delivery'/Path(member.name).parts[0]/Path(member.name).name;p.parent.mkdir(parents=True,exist_ok=True);data=archive.extractfile(member).read()
     if p.suffix=='.csv':data=data.replace(b'\r\n',b'\n')
     p.write_bytes(data)
 (DEST/'jobs.json.gz').write_bytes(gzip.compress(json.dumps(jobs).encode(),mtime=0))
 inventory=[]
 with tarfile.open(DEST/'raw_jobs.tar.gz','w:gz') as archive:
  for n,data in sorted(raw.items()):
   item=tarfile.TarInfo(n);item.size=len(data);item.mtime=0;archive.addfile(item,io.BytesIO(data));inventory.append(dict(path=n,bytes=len(data),sha256=sha(data)))
 with tarfile.open(DEST/'raw_jobs.tar.gz') as archive:
  assert len(archive.getmembers())==len(inventory)
  for expected,item in zip(inventory,archive.getmembers()):assert item.name==expected['path'] and sha(archive.extractfile(item).read())==expected['sha256']
 write(DEST/'raw_manifest.json',inventory)
 summary=dict(at=r.stamp(),ok=True,session=str(ROOT),physical_gpu=4,epoch=0,jobs=407,kinds=dict(collections.Counter(x['kind'] for x in jobs)),qualification_jobs=29,three_round_pairs=46,full_comparisons=357,full_elements=sum(int(x[1]) for x in comps),failed_elements=0,paired_SQTT_captures=10,new_heldout_seed_drawn=False,accepted_stable50=0,accepted_attributable30=0,adopted=False,Goal_complete=False,source24_and_binary='k64_build_isa_audit.json andcompile/build10/completed.json',old_session_unchanged=True)
 write(DEST/'audit_summary.json',summary)
 (DEST/'README.md').write_text('# K64双槽及wave分工：完整发现与负面结论\n\n明确授权的12小时续研，GPU4串行、epoch0、binary d94d9808dede6ae6ba6e8cd332d9e6b91196795e08a96713b13278ccc6a59374。新实验截止北京时间20:10:44，硬截止22:10:44；旧会话期限保留。本阶段完成29资格、81次全输出和guard；46个完整三轮配对，原始日志审计通过。\n\n四wave19代表GM候选/当前CK=1.081196，5条改善>2%、11条回退>2%、12条高CV；八wave14代表GM1.136593，1条改善、10条回退、3条高CV。集合不同，不能直接据两个GM排名。完整域及已登记低G范围都不采用；三轮发现和个体赢家不构成七block验收。稳定50/归因跨桶30计数均为0，没有新heldout或全1193/881测试。原P6k、BLAS与Excel不改。\n\n两项K64配置减少了同步阶段，但寄存器和LDS代价明显。四wave621/805平均驻留约6.36/6.68→3.17/3.63。另预声明补采ID805的K32D4八wave6027/K64D2八wave6029直接控制：同M64/N128、48KiBLDS，实际驻留7.383→7.210，HBM557.6→562.8MB；wait比.170→.256、LDS冲突比.178→.380，SALU1303→680而VMEM194/LDS786/MMAC1024每wave相同，正式三轮慢15.3%。这支持继续区分等待、LDS访问和资源代价，计数本身不证明单一根因。K64四wave与旧6022的控制还耦合MMAC顺序，不称为纯K因果。\n\n第10批4对8份SQTT，补批10b为上述实验K32/K64的1对2份；不是把6027称为当前最佳CK。两包原名/sidecar与XCompute说明保留，下载路径/大小/SHA在两份delivery收据，GUI未本地打开。perf耗时不作正式性能成绩，不把不同wave累计时钟相加。\n\n本阶段407个作业（含补采2）全部终态，357次CK全输出失败0，实际元素数见audit_summary。raw_jobs.tar.gz含原始job记录、原字节过滤的日志、精确runner版本和manifest，每成员SHA核对；大型perf、二进制与厂商ISA不入Git。24份实际编译输入含用户overlay，仅保存SHA并引用现场compile/build10/compiled_sources，不代提交用户修改。自有9源码提交4eb728782b9c9a739a075cf08d157428af224c85（实验用，不合入），用户9tracked/10untracked/readfirstlane保留。\n\nordinary CPU归档helper一次基线不匹配和一次收据git status目录折叠均已保留并独立修复；未影响GPU、原始性能或用户文件。不将这些审计问题归为MKFD。新的N256后续实验单独冻结/计时，不混入本档。\n')
 files=[dict(path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(DEST.rglob('*')) if p.is_file()]
 write(DEST/'manifest.json',files)
 before=git('diff');assert not git('diff','--cached','--name-only').strip()
 index=ROOT/'k64_archive.index';assert not index.exists();env=os.environ.copy();env['GIT_INDEX_FILE']=str(index)
 subprocess.run(['git','read-tree','HEAD'],cwd=REPO,env=env,check=True)
 owned={}
 for p in sorted(DEST.rglob('*')):
  if not p.is_file():continue
  n=str(p.relative_to(REPO));data=p.read_bytes();oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=REPO,input=data).decode().strip();owned[n]=oid
  subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+oid+','+n],cwd=REPO,env=env,check=True)
 assert set(git('diff','--cached','--name-only',env=env).decode().splitlines())==set(owned)
 subprocess.run(['git','diff','--cached','--check'],cwd=REPO,env=env,check=True)
 subprocess.run(['git','commit','-m','P6l K64完整46对与两批SQTT，负结果归档（实验用，不合入）'],cwd=REPO,env=env,check=True)
 head=git('rev-parse','HEAD').decode().strip()
 for n,oid in owned.items():
  assert git('show','HEAD:'+n)==(REPO/n).read_bytes()
  subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+oid+','+n],cwd=REPO,check=True)
 assert git('diff')==before and not git('diff','--cached','--name-only').strip()
 receipt=dict(at=r.stamp(),commit=head,files=len(owned),raw_members=len(inventory),Git_blob_verified=True,user_worktree_preserved=True,push=False,Goal_complete=False)
 write(ROOT/'k64_archive_commit.json',receipt)
 print('CONTINUATION_K64_RAW_AND_GIT_ARCHIVE_PASS',json.dumps(receipt),flush=True)
if __name__=='__main__':main()

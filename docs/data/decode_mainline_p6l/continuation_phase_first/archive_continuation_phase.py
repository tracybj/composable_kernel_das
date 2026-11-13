"""Audit and locally archive the completed, explicitly authorized continuation."""
from pathlib import Path
import collections,datetime,gzip,hashlib,io,json,os,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
DEST=REPO/'docs/data/decode_mainline_p6l/continuation_phase_first'
def read(p):return json.loads(p.read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def git(*a,env=None):return subprocess.check_output(['git',*a],cwd=REPO,env=env)
def main():
 import runtime as r
 session=read(ROOT/'session.json');assert time.time()<session['gpu_deadline_epoch']
 assert read(ROOT/'phase_controller_status.json')['state']=='complete'
 assert read(ROOT/'phase_qualification_audit.json')['ok']
 analysis=read(ROOT/'phase_discovery_analysis.json');assert analysis['complete_phase'] and analysis['completed_pairs']==46
 captures=read(ROOT/'phase_profiles.json');assert len(captures)==10 and all(x['raw_token_bytes']>=4096 and x['active_CU_index']==0 for x in captures)
 jobs=[json.loads(x) for x in (ROOT/'jobs.jsonl').read_text().splitlines()]
 started=[json.loads(x) for x in (ROOT/'started.jsonl').read_text().splitlines()]
 assert len(jobs)==len(started)==405 and all(x['ok'] and x['rc']==0 and x['physical_gpu']==4 and x['epoch']==0 for x in jobs)
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
 assert len(comps)==351
 for (n,s),p in versions.items():data=p.read_bytes();assert sha(data)==s;raw['source_versions/'+p.name]=data
 for s,p in manifests.items():raw['manifests/'+p.name]=p.read_bytes()
 for n in ['gates.jsonl','started.jsonl','kernel_checks.jsonl','phase_controller_events.jsonl']:raw['journals/'+n]=(ROOT/n).read_bytes()
 delivery=read(ROOT/'profile_delivery_batch9.json');package=Path(delivery['path']);assert package.stat().st_size==delivery['bytes'] and sha(package.read_bytes())==delivery['sha256']
 assert not DEST.exists();DEST.mkdir()
 names=['session.json','authorization.json','continuation_bootstrap.json','initial_files_sha256.json','historical_exclusion_inventory.json','historical_final_attempts.json','research_plan.json','heldout_protocol.json','phase_hypothesis.json','phase_discovery_plan.json','phase_discovery_results.json','phase_discovery_analysis.json','phase_discovery_analysis.csv','phase_qualification.json','phase_qualification_audit.json','phase_profile_rule.json','phase_profile_plan.json','phase_profile_attempts.json','phase_profiles.json','profile_delivery_batch9.json','phase_campaign_cpu_selector_audit.json','phase_baseline1193_cpu_audit.json','phase_controller_status.json','phase_build_isa_audit.json','compile/build9/source_sha256.json','compile/build9/completed.json','environment_epoch.json','recovery_total.json','phase_profile_analysis/profile_analysis.json','phase_profile_analysis/h3_static_schedule.json','archive_continuation_phase.py']
 for n in names:
  p=DEST/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 with tarfile.open(package) as archive:
  for member in archive.getmembers():
   if Path(member.name).name in ['XCOMPUTE_README.md','profile_manifest.json','profile_manifest.csv']:
    p=DEST/'profile_delivery'/Path(member.name).name;p.parent.mkdir(exist_ok=True);p.write_bytes(archive.extractfile(member).read())
 (DEST/'jobs.json.gz').write_bytes(gzip.compress(json.dumps(jobs).encode(),mtime=0))
 inventory=[]
 with tarfile.open(DEST/'raw_jobs.tar.gz','w:gz') as archive:
  for n,data in sorted(raw.items()):
   item=tarfile.TarInfo(n);item.size=len(data);item.mtime=0;archive.addfile(item,io.BytesIO(data));inventory.append(dict(path=n,bytes=len(data),sha256=sha(data)))
 with tarfile.open(DEST/'raw_jobs.tar.gz') as archive:
  assert len(archive.getmembers())==len(inventory)
  for expected,item in zip(inventory,archive.getmembers()):assert item.name==expected['path'] and sha(archive.extractfile(item).read())==expected['sha256']
 write(DEST/'raw_manifest.json',inventory)
 summary=dict(at=r.stamp(),ok=True,session=str(ROOT),physical_gpu=4,epoch=0,jobs=405,kinds=dict(collections.Counter(x['kind'] for x in jobs)),qualification_jobs=27,three_round_pairs=46,full_comparisons=351,full_elements=sum(int(x[1]) for x in comps),failed_elements=0,paired_SQTT_captures=10,new_heldout_seed_drawn=False,accepted_stable50=0,accepted_attributable30=0,adopted=False,Goal_complete=False,source22_and_binary='phase_build_isa_audit.json andcompile/build9/completed.json',old_session_unchanged=True)
 write(DEST/'audit_summary.json',summary)
 (DEST/'README.md').write_text('''# 明确授权12小时续研：MMAC提前调度的完整有限验证

新会话2026-11-13北京时间10:10:44开始，20:10:44停止新实验，22:10:44结束；旧session和期限保留。GPU4串行、统一锁、fresh健康与pre/post dmesg通过，无本次恢复，不跨旧会话epoch混合成绩。

27项资格、75次全输出及guard通过；46个完整三轮A/B和原始审计通过。compact19代表GM candidate/base=1.015421，8条改善>2%、4条回退>2%、2条高CV；wide14代表GM1.031505，6条改善、8条回退。两个集合不同，不直接据其GM排序。4组直接旧资源方案/提前MMAC调度控制分别约-0.329/-0.158/-0.726/-0.146%，首组高CV，均不证明≥5%普遍收益。

没有整域采用，没有新七block确认、没有新的heldout种子。当前0/50稳定弱项、0/30归因跨桶保持；不是全1193或881测试。旧P6k、冻结BLAS、原Excel不变。

5对10份SQTT全部CU0有效；下载元数据见profile_delivery_batch9.json，含Source/dispatch18/XCD/CU和Wavefronts说明。包内“正式candidate/base变化”为本阶段三轮代表发现值；模板句“正式成绩来自七block”指后续采用门槛，本阶段尚未完成七block。明确本说明以免将发现值当采用成绩。已交付包字节/SHA保持，Windows GUI未在本机打开。

621/312/933候选平均驻留约5.96/7.65/7.70，不能用wait比例下降或驻留恢复单独证明更快；933正式回退约23%，HBM读取约2.192→2.376GB。1009正式改善约7.1%，HBM基本相同。保留已验证事实和未决机制，不从单个计数推导完整因果。

405个作业全部终态，351次CK全输出比对失败0。raw_jobs.tar.gz及raw_manifest.json逐成员SHA核对，jobs.json.gz为完整原始ledger。既有22份实际编译输入含用户overlay，仅由SHA和旧现场副本引用，不代提交用户改动；大型perf、二进制和厂商ISA不入Git。代码仍为实验用、不合入；本归档只提交本轮自有证据。
''')
 files=[dict(path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(DEST.rglob('*')) if p.is_file()]
 write(DEST/'manifest.json',files)
 before=git('diff');assert not git('diff','--cached','--name-only').strip()
 index=ROOT/'phase_archive.index';assert not index.exists();env=os.environ.copy();env['GIT_INDEX_FILE']=str(index)
 subprocess.run(['git','read-tree','HEAD'],cwd=REPO,env=env,check=True)
 owned={}
 for p in sorted(DEST.rglob('*')):
  if not p.is_file():continue
  n=str(p.relative_to(REPO));data=p.read_bytes();oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=REPO,input=data).decode().strip();owned[n]=oid
  subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+oid+','+n],cwd=REPO,env=env,check=True)
 assert set(git('diff','--cached','--name-only',env=env).decode().splitlines())==set(owned)
 subprocess.run(['git','diff','--cached','--check'],cwd=REPO,env=env,check=True)
 subprocess.run(['git','commit','-m','P6l授权续研：完整46对与第9批SQTT，未采用（实验用，不合入）'],cwd=REPO,env=env,check=True)
 head=git('rev-parse','HEAD').decode().strip()
 for n,oid in owned.items():
  assert git('show','HEAD:'+n)==(REPO/n).read_bytes()
  subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+oid+','+n],cwd=REPO,check=True)
 assert git('diff')==before and not git('diff','--cached','--name-only').strip()
 receipt=dict(at=r.stamp(),commit=head,files=len(owned),raw_members=len(inventory),Git_blob_verified=True,user_worktree_preserved=True,push=False,Goal_complete=False)
 write(ROOT/'phase_archive_commit.json',receipt)
 print('CONTINUATION_PHASE_RAW_AND_GIT_ARCHIVE_PASS',json.dumps(receipt),flush=True)
if __name__=='__main__':main()

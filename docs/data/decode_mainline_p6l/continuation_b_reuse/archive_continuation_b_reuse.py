"""Read-only measurement audit and own-only local evidence archive; no GPU."""
from pathlib import Path
import collections,csv,datetime,gzip,hashlib,io,json,math,os,statistics,subprocess,tarfile,time

ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
DEST=REPO/'docs/data/decode_mainline_p6l/continuation_b_reuse'

def read(p):return json.loads(p.read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def git(*a,env=None):return subprocess.check_output(['git',*a],cwd=REPO,env=env)

def main():
    import runtime as r
    session=read(ROOT/'session.json');assert time.time()<session['deadline_epoch']
    assert read(ROOT/'post_b_reuse_build_status.json')['state']=='complete'
    assert read(ROOT/'b_reuse_controller_status.json')['state']=='complete'
    assert read(ROOT/'b_reuse_qualification_audit.json')['ok']
    assert read(ROOT/'b_reuse_build_isa_audit.json')['ok']
    assert read(ROOT/'b_reuse_source_commit.json')['untracked10_bytes_verified']
    rows=read(ROOT/'b_reuse_discovery_results.json');assert len(rows)==59
    binary=read(ROOT/'compile/build12/completed.json')['benchmark_sha256']
    epoch=read(ROOT/'environment_epoch.json')['id']
    assert all(x['binary_sha256']==binary and x['epoch']==epoch for x in rows)
    captures=read(ROOT/'b_reuse_profiles.json');attempts=read(ROOT/'b_reuse_profile_attempts.json')
    assert len(captures)==2*len(read(ROOT/'b_reuse_profile_plan.json')['pairs'])
    assert all(x['raw_token_bytes']>=4096 and x['epoch']==epoch and x['benchmark_sha256']==binary for x in captures)
    alljobs=[json.loads(x) for x in (ROOT/'jobs.jsonl').read_text().splitlines()]
    allstarted=[json.loads(x)['record'] for x in (ROOT/'started.jsonl').read_text().splitlines()]
    jobs=[x for x in alljobs if x['benchmark_sha256']==binary]
    started=[x for x in allstarted if x['benchmark_sha256']==binary]
    assert len(jobs)==len(started)==38+59*8+len(attempts)
    assert {x['key'] for x in jobs}=={x['key'] for x in started}
    gates={x['key']:x for x in map(json.loads,(ROOT/'gates.jsonl').read_text().splitlines())}
    checks={x['key']:x for x in map(json.loads,(ROOT/'kernel_checks.jsonl').read_text().splitlines())}
    raw={};versions={};manifests={};comparisons=[]
    for x in jobs:
        assert x['ok'] and x['rc']==0 and x['physical_gpu']==4 and x['epoch']==epoch
        assert datetime.datetime.fromisoformat(x['at']).timestamp()<session['gpu_deadline_epoch']
        assert x['environment']['HIP_VISIBLE_DEVICES']==x['environment']['CUDA_VISIBLE_DEVICES']=='4'
        gate=gates[x['key']]
        assert gate['rc']==0 and gate['hcu']=='0.0%' and 'MKFD' not in gate['output'].upper()
        assert gate['mem_available_kib']>=64*1024*1024
        assert all(checks[x['key']+s]['rc']==0 and not checks[x['key']+s]['events'] for s in ['_pre','_post'])
        parsed=r.parse_output((Path(x['path'])/'run.log').read_text(),x['kind'],x.get('expected'),x.get('repeats',1),x.get('regression_kind'))
        if x['kind']=='accuracy':
            assert [list(c) for c in parsed['comparisons']]==x['comparisons']
            assert all(int(c[1])==x['shape']['M']*x['shape']['N']*x['shape']['G'] and int(c[2])==0 for c in parsed['comparisons'])
            comparisons+=parsed['comparisons']
        if x['kind']=='perf':assert parsed['result']==x['result']
        for p in Path(x['path']).iterdir():
            if p.is_file():
                assert p.suffix not in ['.perf','.o','.co','.isa']
                raw['jobs/'+p.parent.name+'/'+p.name]=p.read_bytes()
        for n,s in x['script_sha256'].items():versions[n,s]=ROOT/'source_versions'/(s+'_'+n)
        manifests[x['plan_sha256']]=ROOT/'manifests'/(x['plan_sha256']+'.json')
    assert len(comparisons)==462
    for (n,s),p in versions.items():
        data=p.read_bytes();assert sha(data)==s;raw['source_versions/'+p.name]=data
    for s,p in manifests.items():
        raw['manifests/'+p.name]=p.read_bytes()
        for name,digest in read(p)['runner_sha256'].items():
            if name.startswith('drafts/'):
                data=(ROOT/name).read_bytes();assert sha(data)==digest
                raw['auxiliary_sources/'+name]=data
    keys={x['key'] for x in jobs}
    for name in ['gates.jsonl','started.jsonl','kernel_checks.jsonl']:
        lines=[]
        for line in (ROOT/name).read_bytes().splitlines(keepends=True):
            item=json.loads(line);key=item['record']['key'] if name=='started.jsonl' else item['key']
            if key in keys or (name=='kernel_checks.jsonl' and key.rsplit('_',1)[0] in keys):lines.append(line)
        raw['journals/'+name]=b''.join(lines)
    raw['journals/b_reuse_controller_events.jsonl']=(ROOT/'b_reuse_controller_events.jsonl').read_bytes()
    delivery=read(ROOT/'profile_delivery_batch12.json');package=Path(delivery['path'])
    assert package.stat().st_size==delivery['bytes'] and sha(package.read_bytes())==delivery['sha256']
    assert not DEST.exists();DEST.mkdir()
    names=['session.json','authorization.json','initial_files_sha256.json','historical_exclusion_inventory.json',
           'historical_final_attempts.json','research_plan.json','heldout_protocol.json','b_reuse_mechanism_decision.json',
           'b_reuse_discovery_plan.json','b_reuse_discovery_results.json','b_reuse_discovery_analysis.json',
           'b_reuse_qualification.json','b_reuse_qualification_audit.json','b_reuse_profile_rule.json',
           'b_reuse_profile_plan.json','b_reuse_profile_attempts.json',
           'b_reuse_profiles.json','profile_delivery_batch12.json','b_reuse_campaign_cpu_selector_audit.json',
           'b_reuse_baseline1193_cpu_audit.json','b_reuse_controller_status.json','b_reuse_build_isa_audit.json',
           'b_reuse_source_commit.json','compile/build12/source_sha256.json','compile/build12/completed.json',
           'environment_epoch.json','recovery_total.json','b_reuse_profile_analysis/profile_analysis.json',
           'b_reuse_profile_analysis/h3_static_schedule.json','drafts/b_reuse_factor/cpu_policy/results.json',
           'drafts/b_reuse_factor/cpu_policy/public_disable_equivalence.json','drafts/b_reuse_factor/cpu_model_result.json',
           'drafts/b_reuse_factor/old_partition_compatibility.json',
           'drafts/b_reuse_factor/actual_cfg_proof.json',
           'drafts/b_reuse_factor/probe_result.json',
           'drafts/b_reuse_factor/failed_probe_include_order/reason.json',
           'archive_continuation_b_reuse.py']
    for name in names:
        p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    with tarfile.open(package) as archive:
        for member in archive.getmembers():
            if Path(member.name).name in ['XCOMPUTE_README.md','profile_manifest.json','profile_manifest.csv']:
                p=DEST/'profile_delivery'/Path(member.name).name;p.parent.mkdir(parents=True,exist_ok=True)
                data=archive.extractfile(member).read()
                if p.suffix=='.csv':data=data.replace(b'\r\n',b'\n')
                p.write_bytes(data)
    (DEST/'jobs.json.gz').write_bytes(gzip.compress(json.dumps(jobs).encode(),mtime=0))
    inventory=[]
    with tarfile.open(DEST/'raw_jobs.tar.gz','w:gz') as archive:
        for name,data in sorted(raw.items()):
            item=tarfile.TarInfo(name);item.size=len(data);item.mtime=0
            archive.addfile(item,io.BytesIO(data));inventory.append(dict(path=name,bytes=len(data),sha256=sha(data)))
    with tarfile.open(DEST/'raw_jobs.tar.gz') as archive:
        assert len(archive.getmembers())==len(inventory)
        for expected,item in zip(inventory,archive.getmembers()):
            assert item.name==expected['path'] and sha(archive.extractfile(item).read())==expected['sha256']
    write(DEST/'raw_manifest.json',inventory)
    pairs={x['key']:x for x in read(ROOT/'b_reuse_discovery_plan.json')['discovery_pairs']}
    fields=['pair','ID','M','N','K','G','variant','control','base_instance','candidate_instance',
            'CK_base_ms','CK_candidate_ms','CK_base_TFLOPS','CK_candidate_TFLOPS','frozen_BLAS_ms',
            'frozen_BLAS_TFLOPS','R_base_frozen','R_candidate_frozen','delta_pct','high_cv',
            'base_rounds','candidate_rounds','base_config','candidate_config','epoch','binary_sha256','keys']
    with (DEST/'paired_results_with_tflops.csv').open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            v={k:row[k] for k in fields if k in row};v.update(row['shape']);p=pairs[row['pair']]
            flops=2*math.prod(row['shape'][k] for k in ['M','N','K','G'])
            v.update(CK_base_TFLOPS=flops/row['CK_base_ms']/1e9,
                     CK_candidate_TFLOPS=flops/row['CK_candidate_ms']/1e9,
                     frozen_BLAS_TFLOPS=flops/row['frozen_BLAS_ms']/1e9,
                     R_base_frozen=row['frozen_BLAS_ms']/row['CK_base_ms'],
                     R_candidate_frozen=row['frozen_BLAS_ms']/row['CK_candidate_ms'],
                     base_config=p['base'],candidate_config=p['candidate'])
            for k in ['base_rounds','candidate_rounds','base_config','candidate_config','keys']:
                v[k]=json.dumps(v[k],ensure_ascii=False,separators=(',',':'))
            writer.writerow(v)
    groups={}
    for variant in ['d2','d3']:
        a=[x for x in rows if x['variant']==variant and not x['control']];assert len(a)==19
        groups[variant]=dict(pairs=19,geometric_candidate_over_base=math.exp(statistics.mean(math.log(x['ratio_candidate_over_base']) for x in a)),
                             improve_gt2=sum(x['delta_pct']<-2 for x in a),regress_gt2=sum(x['delta_pct']>2 for x in a),
                             high_cv=sum(x['high_cv'] for x in a))
    summary=dict(at=r.stamp(),ok=True,physical_gpu=4,epoch=epoch,binary_sha256=binary,jobs=len(jobs),
                 kinds=dict(collections.Counter(x['kind'] for x in jobs)),qualification_jobs=38,three_round_pairs=59,
                 full_comparisons=462,full_elements=sum(int(x[1]) for x in comparisons),failed_elements=0,
                 paired_SQTT_captures=len(captures),variants=groups,adopted=False,accepted_stable50=0,
                 accepted_attributable30=0,Goal_complete=False,GUI_verified=False,initial_user_edits_preserved=True)
    write(DEST/'audit_summary.json',summary)
    text=['# B复用 N256 双槽／三槽：完整代表发现与 SQTT','',
          '本阶段为独立 H3 流水的有限 N 面板／预取深度研究，全部经 GPU4 串行、同二进制和同 epoch 完成。',
          f'正式 benchmark SHA256：`{binary}`。38 资格作业、59 个三轮配对、462 次全输出比较及真实 guard 全部通过。','']
    for variant,values in groups.items():text.append(f'- {variant}：19 条代表，耗时几何平均候选／当前 CK {values["geometric_candidate_over_base"]:.6f}；改善超过 2% 共 {values["improve_gt2"]} 条，回退超过 2% 共 {values["regress_gt2"]} 条，高 CV {values["high_cv"]} 条。')
    text+=['','本档仅是完整三轮发现，尚未采用；范围内的全部收益／回退保留，后续候选仍须完整属性域七 block 验收。没有根据个体赢家更新原 P6k、BLAS 或 Excel，目标计数仍为 0/50、0/30。原表 BLAS 来自冻结 GPU0 Native GroupGEMM，不能称为同卡新鲜 BLAS；所有新 CK 对照来自 GPU4 同二进制。',
           '',f'SQTT 共 {len(captures)} 份，下载收据为 profile_delivery_batch12.json；XCompute 打开说明和原始相对路径另附。分工控制 base 为原 N256 的同深度实验核，不能称为当前最佳 CK。GUI 未在本机运行；perf 时长不作正式成绩，wave 累计时钟不能相加充当墙钟。',
           '', '本轮 rule 在捕获前固定：两深度各 19 代表，固定四 shape 和缺失类别，另采 ID805 同深度原分工／B复用直接控制；预取深度相同，生产者不变。',
           '', 'paired_results_with_tflops.csv 保留全部 59 对、配置／实际实例、TFLOPS、三轮各摘要与高 CV；raw_jobs.tar.gz 按成员 SHA 保存全部日志、runner 精确版本和 manifest。大型 perf、实际编译源码中用户 overlay、二进制与厂商 ISA 不入 Git；现场 compile/build12/compiled_sources 保留实际 25 份编译输入。',
           '', '原会话期限保持；本次新增窗口的新实验截止北京时间 20:10:44、硬截止 22:10:44。未跑 1193/881 全量 GPU，没有 push；自有实验源码单独提交，原用户 tracked/untracked 与 readfirstlane 保留。','']
    (DEST/'README.md').write_text('\n'.join(text))
    write(DEST/'manifest.json',[dict(path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(DEST.rglob('*')) if p.is_file()])
    before=git('diff');assert not git('diff','--cached','--name-only').strip()
    index=ROOT/'b_reuse_archive.index';assert not index.exists();env=os.environ.copy();env['GIT_INDEX_FILE']=str(index)
    subprocess.run(['git','read-tree','HEAD'],cwd=REPO,env=env,check=True);owned={}
    for p in sorted(DEST.rglob('*')):
        if not p.is_file():continue
        name=str(p.relative_to(REPO));oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=REPO,input=p.read_bytes()).decode().strip();owned[name]=oid
        subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+oid+','+name],cwd=REPO,env=env,check=True)
    assert set(git('diff','--cached','--name-only',env=env).decode().splitlines())==set(owned)
    subprocess.run(['git','diff','--cached','--check'],cwd=REPO,env=env,check=True)
    subprocess.run(['git','commit','-m','P6l B复用 N256完整59对与SQTT，代表负结果归档（实验用，不合入）'],cwd=REPO,env=env,check=True)
    for name,oid in owned.items():
        assert git('show','HEAD:'+name)==(REPO/name).read_bytes()
        subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+oid+','+name],cwd=REPO,check=True)
    assert git('diff')==before and not git('diff','--cached','--name-only').strip()
    receipt=dict(at=r.stamp(),commit=git('rev-parse','HEAD').decode().strip(),files=len(owned),raw_members=len(inventory),
                 Git_blob_verified=True,user_worktree_preserved=True,push=False,Goal_complete=False)
    write(ROOT/'b_reuse_archive_commit.json',receipt)
    print('CONTINUATION_B复用 N256_RAW_AND_GIT_ARCHIVE_PASS',json.dumps(receipt),flush=True)

if __name__=='__main__':main()

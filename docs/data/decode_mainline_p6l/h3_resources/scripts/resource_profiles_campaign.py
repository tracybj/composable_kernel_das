"""Predeclared bounded resource SQTT comparisons; no GPU on import."""
from pathlib import Path
import argparse,copy,csv,fcntl,json,math,re,shutil,statistics,tarfile,time,uuid
import campaign as c,plan,profile_capture as trace,runtime as r
import profile_h3_factors as old_identities
ROOT=Path(__file__).resolve().parent
RULE=ROOT/'resource_profile_rule.json'
REGPLAN=ROOT/'resource_profile_plan.json'
CAPTURES=ROOT/'resource_profiles.json'
ATTEMPTS=ROOT/'resource_profile_attempts.json'
ANALYSIS=ROOT/'resource_profile_analysis'

class identities:
    @staticmethod
    def kernel_identity(case):
        ident=case['expected'];m=case['M'];rows=1 if m<=16 else 2 if m<=32 else 4
        if ident in [6024,6025]:
            kind='Compact' if ident==6024 else 'Wide'
            return 'DecodeNnCooperative'+kind+'Kernel',f'DecodeNnCooperative{kind}Kernel<{rows}>'
        return old_identities.kernel_identity(case)

def freeze():
    assert time.time()+60<r.read(ROOT/'session.json')['gpu_deadline_epoch']
    site=r.Site(ROOT);assert site.manifest['phase']=='h3_resource_discovery'
    rows=r.read(ROOT/'resource_discovery_results.json');assert len(rows)==46
    assert all(x['binary_sha256']==site.manifest['benchmark_sha256'] and x['epoch']==site.epoch() for x in rows)
    rule=r.read(RULE);actual=[x for x in rows if not x['control']]
    ids=set(x['ID'] for x in actual if x['variant']=='wide')
    def score(name):
        rr=[x['ratio_candidate_over_base'] for x in actual if x['variant']==name and x['ID'] in ids]
        assert len(rr)==14
        return math.exp(statistics.mean(math.log(x) for x in rr)),name
    wide_choice=min(['compact','wide'],key=score)
    selected=[x for x in actual if x['variant']==('compact' if x['shape']['M']<=16 else wide_choice)]
    assert len(selected)==19
    byid={x['ID']:x for x in selected}
    chosen=[byid[i] for i in rule['fixed_ids']]
    def category(x):return 'gain' if x['delta_pct'] < -2 else 'loss' if x['delta_pct'] > 2 else 'flat'
    for cat in ['gain','flat','loss']:
        if any(category(x)==cat for x in chosen):continue
        options=sorted([x for x in selected if category(x)==cat],key=lambda x:(x['delta_pct'],x['ID'],x['variant']))
        if options:chosen.append(options[len(options)//2])
    assert 4<=len(chosen)<=7 and len({x['pair'] for x in chosen})==len(chosen)
    configs={x['key']:x for x in r.read(ROOT/'resource_discovery_plan.json')['discovery_pairs']}
    pairs=[]
    for row in chosen:
        p=copy.deepcopy(configs[row['pair']]);p.update(formal_base_ms=row['CK_base_ms'],formal_candidate_ms=row['CK_candidate_ms'],formal_delta_pct=row['delta_pct'],prior_accuracy_keys=row['keys'][:2],high_cv=row['high_cv'])
        pairs.append(p)
    prepared=dict(epoch=site.epoch(),binary_sha256=site.manifest['benchmark_sha256'],rule_sha256=plan.sha(RULE),discovery_sha256=plan.sha(ROOT/'resource_discovery_results.json'),wide_M_choice=wide_choice,low_M_choice='compact',selection_is_not_final_adoption=True,pairs=pairs)
    prepared['sha256']=plan.digest(prepared)
    if REGPLAN.exists():assert r.read(REGPLAN)==prepared
    else:r.write(REGPLAN,prepared)
    new=copy.deepcopy(site.manifest);new.pop('plan_sha256',None);new['resource_profile_plan_sha256']=prepared['sha256']
    for name in ['resource_profiles_campaign.py',RULE.name,REGPLAN.name]:new['runner_sha256'][name]=plan.sha(ROOT/name)
    new['plan_sha256']=plan.digest(new);r.write(ROOT/'manifests'/(new['plan_sha256']+'.json'),new);r.write(ROOT/'frozen_plan.json',new)
    print('RESOURCE_PROFILE_SELECTION_FROZEN',len(pairs),'pairs',wide_choice,flush=True)

def capture():
    site=r.Site(ROOT);assert site.manifest['phase']=='h3_resource_discovery'
    rule=r.read(RULE);registered=r.read(REGPLAN)
    assert registered['epoch']==site.epoch()
    captures=r.read(CAPTURES,[]);attempts=r.read(ATTEMPTS,[])
    for pair in registered['pairs']:
        old=[x for x in captures if x['pair']==pair['key']]
        if old:
            assert len(old)==2 and {x['role'] for x in old}=={'base','candidate'}
            assert all(x['epoch']==site.epoch() and x['benchmark_sha256']==registered['binary_sha256'] for x in old)
            assert len({x['active_CU_index'] for x in old})==1
            continue
        for cu in rule['target_cu_order']:
            accepted=[];sparse=False
            for role in ['base','candidate']:
                case=pair[role];key=f'h3resource_id{case["ID"]}_{role}_cu{cu}_{uuid.uuid4().hex[:8]}'
                folder=ROOT/'profiles_resources'/key;folder.mkdir(parents=True,exist_ok=False)
                spec=c.ck(case,'perf',key)
                app=[a.replace('--samples=30','--samples=1').replace('--validate=1','--validate=0') for a in spec['command']]
                kernel,expected=identities.kernel_identity(case)
                cmd=[str(c.XPROF),'--enable-sqtt','--enable-raw','--target-se','0',
                     '--target-cu',str(cu),'--dispatches','18','--kernels',kernel,
                     '--sections',rule['sections'],'--output-dir',str(folder),*app]
                spec.update(kind='profile',command=cmd,variant=role+'_resource_diagnosis',
                    purpose='h3_resource_discovery',sqtt=True,capture_timing_is_formal=False,
                    prior_accuracy_keys=pair['prior_accuracy_keys'],profile_rule_sha256=plan.sha(RULE))
                rec=site.execute(spec);inventory=trace.trace_inventory(folder)
                log=Path(rec['path'],'run.log').read_text()
                dispatches=[int(v) for v in re.findall(r'== dispatch kernel \[(\d+)\]',log)]
                assert dispatches==[18],'Wrong dispatch is not a sparse capture'
                details=(folder/'results.txt').read_text()
                actual=re.search(r'kernel-name\((.*?)\), start_ts',details)[1]
                assert expected is None or expected in re.sub(r'\s+','',actual),'Wrong kernel is not sparse'
                assert 'dispatch[18], gpu-id(4)' in details
                valid=inventory['raw_token_bytes']>=rule['minimum_raw_token_bytes']
                attempts.append(dict(at=r.stamp(),pair=pair['key'],role=role,CU_index=cu,
                    job_key=rec['key'],raw_token_bytes=inventory['raw_token_bytes'],
                    accepted_token_gate=valid,inventory=inventory,actual_kernel=actual))
                r.write(ATTEMPTS,attempts)
                accepted.append(dict(ID=case['ID'],pair=pair['key'],role=role,
                    arm=role+'_resource_diagnosis',variant=pair['variant'],
                    shape={z:case[z] for z in ['M','N','K','G']},dtype='BF16',layout='NN',
                    stage='sqtt',epoch=rec['epoch'],physical_gpu=4,key=rec['key'],
                    actual_instance=case['expected'],algorithm_id=None,kernel_filter=kernel,
                    actual_kernel=actual,expected_kernel=expected,dispatches=dispatches,
                    selected_dispatch=18,XCD=0,SE=0,active_CU_index=cu,sqtt=True,
                    raw_log=str(Path(rec['path'])/'run.log'),
                    benchmark_sha256=site.manifest['benchmark_sha256'],binary_sha256=rec['binary_sha256'],
                    collection_command=cmd,environment=rec['environment'],
                    prior_accuracy_keys=pair['prior_accuracy_keys'],plan_sha256=rec['plan_sha256'],
                    source_sha256=site.manifest['worktree_source_sha256'],script_sha256=rec['script_sha256'],
                    formal_timing=False,point_base_ms=pair['formal_base_ms'],
                    point_candidate_ms=pair['formal_candidate_ms'],point_delta_pct=pair['formal_delta_pct'],
                    **inventory))
                if not valid:
                    sparse=True;print('REGRESSION_SPARSE_RETAINED',case['ID'],role,cu,flush=True);break
            if not sparse:
                assert len(accepted)==2;captures+=accepted;r.write(CAPTURES,captures)
                print('REGRESSION_PAIRED_SQTT_PASS',pair['key'],cu,flush=True);break
        else:raise RuntimeError('All three predeclared CU choices sparse; retain and stop')
    assert len(captures)==2*len(registered['pairs'])

def analyze():
    import analyze_profiles as counters,extract_profile_isa as isa,audit_resource_isa as resource_isa
    captures=r.read(CAPTURES);assert len(captures)==2*len(r.read(REGPLAN)['pairs'])
    ANALYSIS.mkdir(exist_ok=True);r.write(ANALYSIS/'profiles.json',captures)
    counters.ROOT=ANALYSIS;counters.main();isa.ROOT=ANALYSIS;isa.main()
    records=r.read(ANALYSIS/'profile_isa.json');proofs=[]
    for item in records:
        match=re.search(r'DecodeNnCooperative(Compact|Wide)KernelILi(\d+)EE',item['symbol'])
        if not match:continue
        proof=resource_isa.static_schedule(item,match[1],int(match[2]));proofs.append(dict(ID=item['ID'],arm=item['arm'],**proof))
        item['layout_backward_candidates']=item['main_loop_candidates'];item['main_loop_candidates']=[proof['main_loop']]
        lo=int(proof['main_loop']['start_pc'],16);hi=int(proof['main_loop']['end_pc'],16)
        actual=resource_isa.schedule.instructions(item['assembly'])
        item['instruction_anchors']=[dict(line=z['line'],text=z['text']) for z in actual if lo<=z['address']<=hi and z['opcode'].startswith(('s_waitcnt','s_barrier','buffer_load','ds_read','v_mmac'))][:30]
    assert len(proofs)==len(captures)//2
    r.write(ANALYSIS/'profile_isa.json',records);r.write(ANALYSIS/'h3_static_schedule.json',dict(kernels=proofs,not_a_timing_proof=True))
    r.write(ANALYSIS/'same_candidate_code_across_G.json',dict(not_assumed=True,note='Different resource factors and M fragments may use different code; each captured instance is independently identified.'))
    print('RESOURCE_COUNTER_ISA_PASS',len(captures),flush=True)

def deliver():
    captures=r.read(CAPTURES);assert len(captures)==2*len(r.read(REGPLAN)['pairs'])
    codes={(x['ID'],x['arm']):x for x in r.read(ANALYSIS/'profile_isa.json')}
    metrics={(x['ID'],x['arm']):x for x in r.read(ANALYSIS/'profile_analysis.json')['rows']}
    target=ROOT/'delivery/batch8_resources';assert not target.exists();target.mkdir(parents=True)
    manifest=[];text=['# 两种有界资源因素与当前CK对照','',
        '全部46个预登记发现配对结束后采集；固定621/805/1009/312并补齐实际存在的收益/平坦/回退类别。候选选择和这些诊断样本均不是最终采用结论。',
        '', '完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。',
        '先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。',
        '重点检验候选的驻留wave是否被LDS/寄存器限制，以及G改变后访存吞吐、LDS冲突和等待是否对应正式回退。平均驻留数本身不是理论上限证明；长wait与不同wave累计Latency不能直接相加当kernel墙钟。',
        '正式成绩来自独立七block，perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。','']
    for cap in captures:
        code=codes[cap['ID'],cap['arm']];metric=metrics[cap['ID'],cap['arm']]
        assert code['unknown_lines']==0 and cap['raw_token_bytes']>=4096
        source=Path(next(x['path'] for x in cap['files'] if x['path'].endswith('.perf'))).parent
        relative=Path('captures')/source.name;shutil.copytree(source,target/relative)
        entry={k:v for k,v in cap.items() if k not in ['files','raw_log']};entry['files']=[]
        entry.update(kernel_name=code['captured_kernel'],machine_sha256=code['sha256'],resources=metric['geometry'])
        for item in cap['files']:
            p=target/relative/Path(item['path']).relative_to(source)
            assert p.stat().st_size==item['bytes'] and plan.sha(p)==item['sha256']
            entry['files'].append(dict(path=str(p.relative_to(target)),bytes=p.stat().st_size,sha256=plan.sha(p)))
        manifest.append(entry);perf=next(x for x in entry['files'] if x['path'].endswith('.perf'))
        text += [f'## ID{cap["ID"]} {cap["role"]}','',
            f'- MNKG {cap["shape"]}；实例{cap["actual_instance"]}；GPU4/epoch{cap["epoch"]}。',
            f'- 文件 `{perf["path"]}`；dispatch18/XCD0/active-CU-index{cap["active_CU_index"]}。',
            f'- kernel `{code["captured_kernel"]}`。',f'- 资源 {metric["geometry"]}。',
            f'- 正式candidate/base变化{cap["point_delta_pct"]:+.3f}%；平均驻留wave/CU {metric.get("mean_resident_waves_per_CU")}；HBM读取 {metric.get("hbm_read_bytes")}字节；wait/wavecycle {metric.get("wait_any_over_wave_cycles")}。',
            f'- 主循环PC `{code["main_loop_candidates"]}`。','']
        text += ['  指令PC：`'+z['text']+'`' for z in code['instruction_anchors'][:16]]
    r.write(target/'profile_manifest.json',manifest)
    fields=['ID','M','N','K','G','role','actual_instance','physical_gpu','epoch','selected_dispatch','active_CU_index','raw_token_bytes','perf_path','perf_sha256']
    with (target/'profile_manifest.csv').open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,fields);writer.writeheader()
        for item in manifest:
            perf=next(x for x in item['files'] if x['path'].endswith('.perf'))
            row={k:item.get(k) for k in fields};row.update(item['shape']);row.update(perf_path=perf['path'],perf_sha256=perf['sha256']);writer.writerow(row)
    for p in [RULE,REGPLAN,ATTEMPTS,ANALYSIS/'profile_analysis.json',ANALYSIS/'h3_static_schedule.json',ANALYSIS/'same_candidate_code_across_G.json']:
        shutil.copy2(p,target/p.name)
    (target/'XCOMPUTE_README.md').write_text('\n'.join(text)+'\n')
    package=target.parent/'P6l_batch8_resource_factors_SQTT.tar.gz';assert not package.exists()
    with tarfile.open(package,'w:gz') as archive:archive.add(target,arcname=target.name)
    r.write(ROOT/'profile_delivery_batch8.json',dict(at=r.stamp(),path=str(package),bytes=package.stat().st_size,sha256=plan.sha(package),captures=len(captures),gui_verified=False,local_integrity_checked=True))
    print('REGRESSION_PROFILE_DELIVERY_PASS',package,package.stat().st_size,plan.sha(package),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','capture','analyze','deliver']);args=ap.parse_args();r.no_core()
    if args.stage in ['freeze','capture']:
        with r.locked():freeze() if args.stage=='freeze' else capture()
    elif args.stage=='analyze':analyze()
    else:deliver()

if __name__=='__main__':main()

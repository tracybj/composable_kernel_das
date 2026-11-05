#!/usr/bin/env python3
"""Offline best-observed CK summary; never execute a GPU command or change selection."""
import collections,csv,datetime,gzip,hashlib,json,math,re,statistics,subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
V=Path('/workspace/v7_coverage')
REPO=Path('/workspace/composable_kernel')
SHAPES=Path('/workspace/shapes_1178.csv')
CANONICAL='card4_device_args_single_launch_warmup10_samples30_args0'
LABELS={4003:'v4_fallback_N_tail',4002:'v4_64x128x64',4000:'v4_fallback',1000:'bandwidth_family',
 5000:'v7_256x256x16_M01_1',5002:'v7_256x256x16_M01_8',5003:'smallM_64x128x16',
 5004:'smallM_64x64x16',5005:'v7_256x256x16_NoSLC_M01_1',5006:'v7_256x256x16_NoSLC_M01_8',
 5007:'smallM_64x128x16_drain',5008:'smallM_64x128x32',5009:'smallM_128x128x16_square',5010:'smallM_64x64x32'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fields(text):return dict(re.findall(r'([A-Za-z_][\w-]*)=([^\s]+)',text))
def load_csv(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def med(values):return statistics.median(values)
def stamp():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def csv_out(path,rows):
    assert rows
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)
def gz_out(path,rows):
    with gzip.open(path,'wt',encoding='utf-8') as output:
        for row in rows:output.write(json.dumps(row,ensure_ascii=False)+'\n')

def quality(candidate,choices):
    runner=next((row for row in sorted(choices,key=lambda row:row['ms']) if row['scheme']!=candidate['scheme']),None)
    gap=100*(runner['ms']/candidate['ms']-1) if runner else None
    notes=[]
    if runner is None:notes.append('only_one_scheme_measured')
    elif gap<=2:notes.append('near_tie_within2pct')
    if candidate['rounds']<3:notes.append('single_series_no3round_confirmation')
    if not candidate['sample_cv_recorded']:notes.append('sample_CV_not_fully_recorded')
    if candidate['max_sample_cv'] is not None and candidate['max_sample_cv']>.05 or candidate['round_span_pct']>5:notes.append('high_variation')
    if runner:
        spread=max(2,200*(candidate['max_sample_cv'] or 0),200*(runner['max_sample_cv'] or 0),candidate['round_span_pct'],runner['round_span_pct'])
        if gap<=spread:notes.append('gap_within_reported_variation')
        if not runner['sample_cv_recorded']:notes.append('runner_sample_CV_not_fully_recorded')
    if any('Legacy MKFD/' in note for note in candidate['environment_notes']):notes.append('legacy_environment_warning')
    if not candidate['current_available']:notes.append('historical_source_variant_not_current_switch')
    return ';'.join(notes) or 'clear_numerical_lead_low_reported_variation_not_paired_proof',runner,gap

class Collection:
    def __init__(self):
        self.shapes={int(row['ID']):{**row,**{key:int(row[key]) for key in ['M','N','K','G']}} for row in load_csv(SHAPES)}
        self.by_shape={tuple(row[key] for key in ['M','N','K','G']):i for i,row in self.shapes.items()}
        assert len(self.shapes)==len(self.by_shape)==1193
        self.rows=[];self.excluded=[];self.catalog={};self.remaps=[]
        self.pin(SHAPES,'shape input')
    def pin(self,path,note=''):
        path=Path(path);key=str(path)
        if key not in self.catalog:self.catalog[key]=dict(path=key,sha256=digest(path),bytes=path.stat().st_size,note=note,accepted=0,excluded=0)
        return key
    def reject(self,source,key,reason,**extra):
        self.excluded.append(dict(source=source,key=key,reason=reason,**extra))
        self.catalog[source]['excluded']+=1
    def add(self,source,key,shape,instance,ms,round=1,campaign='',cv=None,scheme=None,revision='current',
            gpu=4,protocol=CANONICAL,environment_note='',validation='CPU_sample_PASS',binary_sha='',at='',label_id=None):
        i=self.by_shape.get(tuple(shape))
        if i is None:return self.reject(source,key,'MNKG_not_in_active1193',shape=shape)
        if label_id is not None and int(label_id)!=i:
            self.remaps.append(dict(source=source,key=key,label_id=int(label_id),actual_id=i,actual_MNKG=shape))
        if not math.isfinite(ms) or ms<=0:return self.reject(source,key,'invalid_time')
        scheme=scheme or LABELS.get(instance,'instance_'+str(instance))
        if revision!='current' and instance==5000:scheme=revision+'_'+scheme
        available=revision=='current' and scheme!='historical_v7_M01_4'
        current_id=5002 if scheme==LABELS[5002] else instance
        if scheme==LABELS[5002] and shape[2]<=4096:available=False
        if scheme=='historical_v7_M01_4':current_id=None
        row=dict(observation_id=len(self.rows)+1,ID=i,M=shape[0],N=shape[1],K=shape[2],G=shape[3],
            scheme=scheme,recorded_instance_id=instance,current_instance_id=current_id,current_available=available,
            ms=ms,round=str(round),campaign=campaign,sample_cv=cv,gpu=gpu,protocol=protocol,
            source=source,source_key=key,source_sha256=self.catalog[source]['sha256'],revision=revision,
            environment_note=environment_note,validation=validation,binary_sha256=binary_sha,measured_at=at)
        self.rows.append(row);self.catalog[source]['accepted']+=1
    def command_shape(self,cmd):
        params=fields(' '.join(cmd));return tuple(int(params[key]) for key in ['m','n','k','groups'])
    def full(self,directory,note):
        path=V/directory/'perf/summary.txt';source=self.pin(path,note)
        for n,line in enumerate(path.read_text().splitlines(),1):
            f=line.split();p=fields(line)
            if 'result=PASS rc=0' not in line or 'failed=0' not in line:
                self.reject(source,str(n),'failed_or_incomplete',line=line);continue
            self.add(source,str(n),tuple(map(int,f[1:5])),int(p['inst']),float(p['ms']),campaign=directory,
                cv=float(p['cv']),at='2026-10-30' if directory=='full1178' else '2026-11-03',
                environment_note=note,validation='full_output_same_shape_PASS' if directory=='full1178' else 'CPU_sample_PASS',label_id=int(f[0]))
    def legacy_logs(self,directory,stage,revision_fn):
        for path in sorted((V/directory).glob('*.log')):
            text=path.read_text(errors='replace')
            if 'PERF backend=ck_device_args' not in text:continue
            source=self.pin(path,'legacy formal benchmark; join by CONFIG MNKG, not filename ID')
            config=re.search(r'^CONFIG (.+)$',text,re.M)
            if not config:self.reject(source,path.name,'missing_CONFIG_cannot_identify_shape');continue
            cfg=fields(config[1]);perf=re.search(r'^PERF backend=ck_device_args (.+)$',text,re.M)
            if not (cfg.get('dtype')=='bf16' and cfg.get('layout')=='NN' and cfg.get('variable_axis')=='none' and cfg.get('num_cu')=='0' and cfg.get('warmup')=='10' and cfg.get('samples')=='30' and cfg.get('include_args_build')=='0'):
                self.reject(source,path.name,'different_contract_or_timing',CONFIG=cfg);continue
            if 'SUMMARY result=PASS' not in text or not re.search(r'ACCURACY backend=ck repeat=0 .*?failed=0(?:\s|$)',text):
                self.reject(source,path.name,'precision_or_execution_not_PASS');continue
            value=fields(perf[1]);sel=re.search(r'SELECT instance_id=(\d+)',text)
            assert sel
            shape=tuple(int(cfg[key]) for key in ['m','n','k','groups'])
            revision=revision_fn(path.name)
            if revision=='legacy_P1' and shape[0]%256:
                self.reject(source,path.name,'known_unsafe_pre_rowmask_M_tail');continue
            rnd=re.search(r'_r(\d+)',path.stem);context=re.sub(r'_r\d+','',path.stem)
            # Keep independent before/after/control series separate, never take one fastest round.
            context=re.sub(r'^id\d+_','',context)
            label=re.search(r'id(\d+)',path.name)
            self.add(source,path.name,shape,int(sel[1]),float(value['median_ms']),round=rnd[1] if rnd else 1,
                campaign=stage+'/'+context,cv=float(value['cv']),revision=revision,label_id=int(label[1]) if label else None)
    def text_timing(self,path,campaign,shape_map=None,mode='standard',gpu=4):
        source=self.pin(path,'legacy aggregate text; preserve successful round medians')
        for n,line in enumerate(Path(path).read_text().splitlines(),1):
            f=fields(line)
            if 'median_ms' not in f:continue
            if f.get('rc')!='0' or not re.search(r'failed=0(?:\s|$)',line):
                self.reject(source,str(n),'failed_or_incomplete',line=line);continue
            label=int(f['id']) if 'id' in f else None
            shape=tuple(int(f[k]) for k in ['M','N','K','G']) if 'M' in f else shape_map[label] if shape_map else tuple(self.shapes[label][k] for k in ['M','N','K','G'])
            scheme=None;revision='current';instance=int(f['instance_id'])
            if mode=='order':
                if f['v']=='m01_4':scheme='historical_v7_M01_4'
                elif f['v']=='m01_8':scheme=LABELS[5002]
            if mode=='grid' and f['v']=='m01_8':scheme=LABELS[5002]
            key=f'{n}';cv=None
            if mode=='order':
                log=Path(path).parent/(f'{f["v"]}_id{label}_r{f["r"]}.log')
                if log.exists():
                    self.pin(log,'CONFIG/CV cross-check for order experiment')
                    text=log.read_text();cfg=re.search(r'^CONFIG (.+)$',text,re.M)
                    if cfg:
                        values=fields(cfg[1]);shape=tuple(int(values[k]) for k in ['m','n','k','groups'])
                    perf=re.search(r'^PERF backend=ck_device_args (.+)$',text,re.M)
                    if perf and float(fields(perf[1])['median_ms'])==float(f['median_ms']):cv=float(fields(perf[1])['cv'])
            self.add(source,key,shape,instance,float(f['median_ms']),round=f['r'],campaign=campaign+'/'+f.get('set','all')+'/'+f['v'],
                cv=cv,scheme=scheme,revision=revision,gpu=gpu,label_id=label,
                protocol=CANONICAL if gpu==4 else 'card0_device_args_single_launch_warmup10_samples30_args0',
                environment_note='Legacy round records; CV unavailable where raw log not retained')
    def journal(self,directory):
        path=V/directory/'jobs.jsonl';source=self.pin(path,'raw journal; successful formal timings only')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        invalid=set()
        if directory=='p6e':
            inv=V/directory/'invalidations.json';self.pin(inv,'mandatory environment invalidations')
            invalid=set(json.loads(inv.read_text())['raw_keys'])
        # Match accepted canonical records, not the fastest retry.
        canonical={}
        for row in rows:
            if row.get('ok') and row['key'] not in invalid:canonical[row.get('logical_key',row['key'])]=row
        for row in rows:
            key=row.get('logical_key',row['key'])
            timed=(row.get('kind') in ['perf','native_reference_ck'] or row.get('mode')=='timing' and row.get('backend')!='blas' or row.get('suite')=='timing' and row.get('backend')=='ck' or row.get('timing') and not row.get('profile'))
            if not timed:continue
            if row['key'] in invalid:
                self.reject(source,row['key'],'explicit_environment_invalidation',result=row.get('result'));continue
            if not row.get('ok') or canonical.get(key) is not row:
                self.reject(source,row['key'],'failed_or_superseded_protocol_attempt',result=row.get('result'));continue
            cmd=row['command'];params=fields(' '.join(cmd))
            if params.get('warmup')!='10' or params.get('samples')!='30' or params.get('include-args-build')!='0':
                self.reject(source,key,'different_timing_contract');continue
            shape=self.command_shape(cmd)
            if directory=='p6':instance=int(row['result']['instance_id'])
            else:
                selected=row.get('selected',row.get('expected'))
                if selected is None:
                    log=Path(row['path'])/'run.log';self.pin(log,'Read actual SELECT for timing journal without selected field')
                    selected=int(re.search(r'SELECT instance_id=(\d+)',log.read_text())[1])
                instance=int(selected)
            group=re.sub(r'(?:^|_)r\d+(?=_|$)','',key);group=re.sub(r'_id\d+(?=_|$)','',group)
            rnd=row.get('round') or (re.search(r'(?:^|_)r(\d+)(?:_|$)',key)[1] if re.search(r'(?:^|_)r(\d+)(?:_|$)',key) else 1)
            self.add(source,row['key'],shape,instance,float(row['result']['median_ms']),round=rnd,campaign=directory+'/'+group,
                cv=row['result'].get('cv'),binary_sha=row.get('benchmark_sha256',''),at=row.get('at',''),
                environment_note='Accepted canonical record; environment_epoch='+str(row.get('environment_epoch','not_recorded')),
                label_id=row.get('id') if row.get('id',0)>0 else None)
    def schemes(self,scheme,i):
        c=self.shapes[i];base={'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN':'1'}
        if scheme.startswith('legacy_') or scheme=='historical_v7_M01_4':return dict(available=False,env={},note='Historical source variant; rebuild recorded source/configuration. Not a current selector switch.')
        for instance,name in LABELS.items():
            if name!=scheme:continue
            if instance==5000:base['CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8']='1'
            elif instance==5002:
                if c['K']<=4096:return dict(available=False,env={},note='M01=8 measured historically below current K>4096 selector gate; needs integration/rebuild')
            elif instance in [5003,5004]:base['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M']=str(128 if instance==5003 else 64)
            elif instance in [5005,5006]:
                base['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC']='1'
                base['CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8']=str(int(instance==5005))
            elif instance>=5007:base['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT']=str({5007:1,5008:2,5009:3,5010:4}[instance])
            elif instance!=4003:base={'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN':'0'}
            return dict(available=True,env=base,note='Use a clean experiment environment; other experimental enable/disable switches unset. Measured opt-in, not a production recommendation.')
        return dict(available=False,env={},note='Unknown historical instance')

def main():
    c=Collection()
    c.full('full1178','Healthy historical snapshot; same-shape full-output pass available')
    c.full('full1178_p4b','Legacy MKFD/re-registration events and high-variance rows; recovery protocol weaker than P6e; historical observation, needs replay if selected')
    # Earlier correctness-fixed implementations and real default fallback routes.
    c.legacy_logs('p1','P1',lambda name:'legacy_P1')
    c.legacy_logs('p2/perf','P2',lambda name:'legacy_P1' if '_p1_' in name else 'legacy_P2')
    c.legacy_logs('p3a/m1','P3a',lambda name:'legacy_P2' if 'v7g16' in name else 'current')
    c.legacy_logs('p3b/perf16','P3b',lambda name:'legacy_P2' if '_p2_' in name else 'current')
    # Coverage summaries include actual MNKG, avoiding historical ID-label mistakes.
    path=V/'p3b/cov/summary_merged.txt';source=c.pin(path,'P3b204-case coverage; actual MNKG join')
    for n,line in enumerate(path.read_text().splitlines(),1):
        f=line.split();p=fields(line)
        if 'result=PASS rc=0' not in line or p.get('acc_failed')!='0':continue
        c.add(source,str(n),tuple(map(int,f[1:5])),int(p['inst']),float(p['ms']),campaign='P3b_coverage',cv=float(p['cv']),label_id=int(f[0]))
    c.text_timing(V/'p4a/order/timing_6.txt','P4a_order6',mode='order')
    # timing.txt is byte-identical to timing_6.txt, not another measurement series.
    duplicate=V/'p4a/order/timing.txt';c.pin(duplicate,'EXCLUDED duplicate of timing_6.txt; exact SHA equal')
    assert digest(duplicate)==digest(V/'p4a/order/timing_6.txt')
    c.text_timing(V/'p4a/order/timing_wide.txt','P4a_order_wide',mode='order')
    c.legacy_logs('p4a/kscan','P4a_Kscan',lambda name:'current')
    c.text_timing(V/'p4b/grid/timing.txt','P4b_grid',mode='grid')
    shape_map={int(f[0]):tuple(map(int,f[1:5])) for f in (line.split() for line in (V/'p4b/ab_shapes.txt').read_text().splitlines())}
    c.pin(V/'p4b/ab_shapes.txt','P4b actual command shape list')
    c.text_timing(V/'p4b/ab/timing.txt','P4b_AB',shape_map=shape_map)
    c.text_timing(V/'p4b/ab/timing_card0_20261102_final.txt','P4b_card0_reference',shape_map=shape_map,gpu=0)
    for name in ['timing.txt','timing2.txt','timing3.txt']:
        c.text_timing(V/'full1178_p4b/retest'/name,'P4b_retest_'+name)
    for directory in ['p6','p6b/stage_wait','p6c','p6d','p6e']:c.journal(directory)
    # Fused20 numbers are retained but never compete against single-launch timings.
    old=Path('/workspace/bf16_nn_customer_benchmark/full_run/results.csv');source=c.pin(old,'fused_test continuous20 historical reference; excluded from primary comparison')
    for n,row in enumerate(load_csv(old),2):
        if row['status']!='ok' or row['accuracy']!='PASS' or not row['CK_ms']:
            c.reject(source,str(n),'failed_historical_fused_run',ID=row['ID']);continue
        c.add(source,str(n),tuple(int(row[k]) for k in ['M','N','K','G']),int(row['CK_instance']),float(row['CK_ms']),campaign='customer_fullrun_fused20',protocol='card4_fused_test_continuous20',label_id=int(row['ID']))
    unknown=Path('/workspace/results_bfnn.csv');source=c.pin(unknown,'Original supplied CK CSV: GPU/timing/instance/accuracy not recorded, reference only')
    c.pin('/workspace/bf16_nn_blas_compare/ck_csv_vs_groupgemm/comparison.csv','Derived comparison view of original supplied CSV; not independent CK measurements')
    unknown_rows=[]
    for n,row in enumerate(load_csv(unknown),2):
        i=int(row['ID']);shape=c.shapes[i]
        assert all(int(row[k])==shape[k] for k in ['M','N','K']) and int(row['E'])==shape['G']
        c.reject(source,str(n),'unknown_protocol_gpu_instance_or_precision_metadata',ID=i,CK_ms=row['CK ms'])
        unknown_rows.append(dict(ID=i,**{key:shape[key] for key in ['M','N','K','G','phase']},CK_ms=row['CK ms'],
            original_BLAS_ms=row['BLAS ms'],gpu='not_recorded',protocol='not_recorded',instance_id='not_recorded',validation='not_recorded',
            source=source,source_line=n,note='Original CSV E validated equal to active G for all1178; excluded from primary best choice'))
    csv_out(ROOT/'unknown_protocol_ck_reference.csv',unknown_rows)
    grouped=collections.defaultdict(list)
    for row in c.rows:grouped[(row['ID'],row['scheme'],row['campaign'],row['gpu'],row['protocol'])].append(row)
    candidates=[]
    for (i,scheme,campaign,gpu,protocol),rows in sorted(grouped.items()):
        by_round=collections.defaultdict(list)
        for row in rows:by_round[row['round']].append(row)
        rounds=sorted(by_round,key=lambda x:int(x));times=[med(row['ms'] for row in by_round[r]) for r in rounds]
        cvs=[row['sample_cv'] for row in rows if row['sample_cv'] is not None]
        recipe=c.schemes(scheme,i)
        candidates.append(dict(candidate_id='C'+str(len(candidates)+1).zfill(6),ID=i,scheme=scheme,campaign=campaign,gpu=gpu,protocol=protocol,
            ms=med(times),rounds=len(rounds),round_labels=rounds,round_median_ms=times,observation_ids=[row['observation_id'] for row in rows],
            max_sample_cv=max(cvs) if cvs else None,round_span_pct=100*(max(times)/min(times)-1),
            sample_cv_recorded=len(cvs)==len(rows),recorded_instance_ids=sorted({row['recorded_instance_id'] for row in rows}),
            current_instance_id=rows[0]['current_instance_id'],current_available=recipe['available'],enable_env=recipe['env'],availability_note=recipe['note'],
            sources=sorted({row['source'] for row in rows}),source_keys=[row['source_key'] for row in rows],
            binary_sha256=sorted({row['binary_sha256'] for row in rows if row['binary_sha256']}),
            environment_notes=sorted({row['environment_note'] for row in rows}),validation=sorted({row['validation'] for row in rows})))
    primary=[row for row in candidates if row['gpu']==4 and row['protocol']==CANONICAL]
    other=[row for row in candidates if row not in primary]
    bs_path=Path('/workspace/bf16NN-groupgemm.csv');c.pin(bs_path,'Frozen BLAS1178 baseline, physicalcard0 repeat20')
    blas={int(row['ID']):dict(ms=float(row['BLAS ms']),gpu=0,source=str(bs_path),sample_span_pct=float(row['sample_span_pct'])) for row in load_csv(bs_path)}
    new_path=V/'p6c/native_new15.json';c.pin(new_path,'Frozen new15 BLAS, physicalcard4 repeat20')
    blas.update({row['id']:dict(ms=row['time_ms'],gpu=4,source=str(new_path),sample_span_pct=row.get('sample_span_pct')) for row in json.loads(new_path.read_text())})
    reference_path=V/'p6c/merged1193.csv';c.pin(reference_path,'Reference original CK columns only; not reimported as duplicate observations')
    original={int(row['ID']):row for row in load_csv(reference_path)}
    out=[];typical=[];current_out=[]
    for i,shape in sorted(c.shapes.items()):
        choices=sorted((row for row in primary if row['ID']==i),key=lambda row:(row['ms'],row['candidate_id']))
        assert choices,('missing shape',i)
        best=choices[0];by_scheme=collections.defaultdict(list)
        for row in choices:by_scheme[row['scheme']].append(row)
        scheme_bests=sorted((min(rows,key=lambda row:row['ms']) for rows in by_scheme.values()),key=lambda row:(row['ms'],row['scheme']))
        quality_text,runner,gap=quality(best,choices)
        runnable=min((row for row in choices if row['current_available']),key=lambda row:row['ms'])
        pooled={scheme:med(row['ms'] for row in rows) for scheme,rows in by_scheme.items()}
        typical_scheme=min(pooled,key=lambda scheme:(pooled[scheme],scheme))
        b=blas[i];r0=float(original[i]['CK_before_ms'])
        out.append(dict(ID=i,**{key:shape[key] for key in ['M','N','K','G','phase']},best_scheme=best['scheme'],
            best_instance_id=best['current_instance_id'],best_recorded_instance_ids=json.dumps(best['recorded_instance_ids']),best_ms=best['ms'],
            BLAS_ms=b['ms'],R_frozen=b['ms']/best['ms'],CK_gpu=4,CK_protocol=CANONICAL,BLAS_gpu=b['gpu'],BLAS_protocol='frozen_native_groupedgemm_repeat20',BLAS_source=b['source'],BLAS_sample_span_pct=b['sample_span_pct'],BLAS_high_span_warning=b['sample_span_pct'] is not None and b['sample_span_pct']>50,
            quality=quality_text,best_rounds=best['rounds'],best_max_sample_cv=best['max_sample_cv'],best_round_span_pct=best['round_span_pct'],
            next_scheme=runner['scheme'] if runner else '',next_scheme_ms=runner['ms'] if runner else None,gap_to_next_scheme_pct=gap,
            within2pct_schemes=';'.join(row['scheme'] for row in scheme_bests if row['ms']<=best['ms']*1.02),
            measured_scheme_count=len(by_scheme),candidate_series_count=len(choices),best_candidate_id=best['candidate_id'],best_campaign=best['campaign'],
            best_source=';'.join(best['sources']),best_source_keys=json.dumps(best['source_keys']),best_binary_sha256=';'.join(best['binary_sha256']),
            current_switch_available=best['current_available'],enable_env=json.dumps(best['enable_env'],sort_keys=True),availability_note=best['availability_note'],
            best_current_scheme=runnable['scheme'],best_current_ms=runnable['ms'],best_current_candidate_id=runnable['candidate_id'],
            typical_best_scheme=typical_scheme,typical_best_ms=pooled[typical_scheme],best_scheme_typical_ms=pooled[best['scheme']],
            original_reference_ms=r0,observed_delta_vs_reference_pct=100*(best['ms']/r0-1),reference_source=str(reference_path),
            comparison_note='Historical best of complete series statistics, not fastest individual round/sample; cross-campaign reference, not a new full sweep or implemented per-ID selector'))
        typical.append(dict(ID=i,scheme=typical_scheme,ms=pooled[typical_scheme],R_frozen=b['ms']/pooled[typical_scheme]))
        current_out.append(dict(ID=i,**{key:shape[key] for key in ['M','N','K','G','phase']},
            best_scheme=runnable['scheme'],best_instance_id=runnable['current_instance_id'],best_ms=runnable['ms'],
            BLAS_ms=b['ms'],R_frozen=b['ms']/runnable['ms'],CK_gpu=4,CK_protocol=CANONICAL,BLAS_gpu=b['gpu'],BLAS_protocol='frozen_native_groupedgemm_repeat20',
            candidate_id=runnable['candidate_id'],source=';'.join(runnable['sources']),source_keys=json.dumps(runnable['source_keys']),
            enable_env=json.dumps(runnable['enable_env'],sort_keys=True),rounds=runnable['rounds'],max_sample_cv=runnable['max_sample_cv'],round_span_pct=runnable['round_span_pct'],
            quality=quality(runnable,[row for row in choices if row['current_available']])[0],
            historical_best_scheme=best['scheme'],historical_best_ms=best['ms'],historical_best_currently_available=best['current_available'],
            note='Current selector switches available; per-shape measured envelope only, no routing policy changed'))
    assert [row['ID'] for row in out]==list(range(1,1194))
    csv_out(ROOT/'best_known_1193.csv',out)
    csv_out(ROOT/'best_current_1193.csv',current_out)
    (ROOT/'best_known_1193.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    csv_out(ROOT/'best_typical_1193.csv',typical)
    flat=[{key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value for key,value in row.items()} for row in primary]
    csv_out(ROOT/'all_candidate_series.csv',flat)
    gz_out(ROOT/'all_candidate_series.jsonl.gz',candidates)
    gz_out(ROOT/'all_observations.jsonl.gz',c.rows)
    gz_out(ROOT/'excluded_records.jsonl.gz',c.excluded)
    csv_out(ROOT/'other_protocol_reference.csv',[{key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value for key,value in row.items()} for row in other])
    (ROOT/'source_catalog.json').write_text(json.dumps(list(c.catalog.values()),ensure_ascii=False,indent=2))
    csv_out(ROOT/'source_catalog.csv',list(c.catalog.values()))
    (ROOT/'shape_label_remaps.json').write_text(json.dumps(c.remaps,indent=2))
    buckets=[]
    for phase in ['prefill','decode','all']:
        rr=[row for row in out if phase=='all' or row['phase']==phase];ratios=[row['R_frozen'] for row in rr]
        buckets.append(dict(phase=phase,total=len(rr),R_ge099=sum(r>=.99 for r in ratios),R_08_099=sum(.8<=r<.99 for r in ratios),R_05_08=sum(.5<=r<.8 for r in ratios),R_03_05=sum(.3<=r<.5 for r in ratios),R_02_03=sum(.2<=r<.3 for r in ratios),R_lt02=sum(r<.2 for r in ratios)))
    csv_out(ROOT/'buckets.csv',buckets)
    summary=dict(created_at=stamp(),cpu_only=True,no_new_gpu_tests=True,shape_count=len(out),phase_counts=dict(collections.Counter(row['phase'] for row in out)),
        source_files=len(c.catalog),observations=len(c.rows),primary_observations=sum(row['gpu']==4 and row['protocol']==CANONICAL for row in c.rows),
        primary_candidate_series=len(primary),other_protocol_series=len(other),excluded_records=len(c.excluded),
        winners_by_scheme=dict(collections.Counter(row['best_scheme'] for row in out)),winners_by_campaign=dict(collections.Counter(row['best_campaign'] for row in out)),
        winner_quality_counts=dict(collections.Counter(flag for row in out for flag in row['quality'].split(';'))),
        historical_winners_unavailable=sum(not row['current_switch_available'] for row in out),
        observed_vs_typical_scheme_disagreements=sum(row['best_scheme']!=row['typical_best_scheme'] for row in out),shape_label_remaps=len(c.remaps),buckets=buckets,
        definition='Minimum campaign/contrast series median across measured compatible schemes; every individual run remains in ledger. Each round is already a30-sample median. No minimum individual sample/round. Typical is median of campaign-series medians per scheme, then minimum scheme.',
        limitations=['Historical envelope across sessions, not proof of a production routing policy or a newly measured1193 sweep','BLAS frozen repeat20 differs from CK single launch; old BLAS GPU0 differs from CK GPU4','No unmeasured shapes/configurations are inferred, including shape-specific kernel combinations','P6e explicit33 invalidations excluded; older environment/variance limitations retained and flagged','Numerical winner within2% or high variance is not a confirmed timing advantage'])
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('BEST_KNOWN_BUILD_PASS',json.dumps({key:summary[key] for key in ['shape_count','observations','primary_candidate_series','other_protocol_series','excluded_records','shape_label_remaps','winners_by_scheme']},ensure_ascii=False))

if __name__=='__main__':main()

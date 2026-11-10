"""Frozen Depth8 contract accuracy and six same-binary A/B comparisons."""
import copy
import json
import statistics
import subprocess
import uuid
from pathlib import Path
import plan
import runtime as r


def prepare(table, output):
    output=Path(output)
    env={plan.V7:'1',plan.V7+'_SMALL_M':'128',plan.DEPTH8:'1'}
    accuracy=[dict(ID=-3100-m,M=m,N=256,K=512,G=64,env=dict(env),expected=6003,
                   accuracy_repeats=3) for m in range(1,33)]
    plan.selector_check(dict(cases=accuracy),output)
    dimensions=[(m,256,512,64) for m in [33,64,128]]+[
        (1,128,512,64),(1,248,512,64),(1,384,512,64),
        (1,256,256,64),(1,256,640,64),(1,256,512,63),(1,256,512,65)]
    boundaries=[dict(ID=-4100-i,**dict(zip(['M','N','K','G'],dims)),env=dict(env),expected=0)
                for i,dims in enumerate(dimensions)]
    def actual(cases):
        lines=[' '.join(map(str,[x['ID'],x['M'],x['N'],x['K'],x['G'],0]+plan.policy_fields(x))) for x in cases]
        p=subprocess.run([str(output/'selector_probe')],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True,timeout=30)
        values=[tuple(map(int,x.split())) for x in p.stdout.splitlines()]
        if len(values)!=len(cases):raise ValueError('Boundary probe count mismatch')
        return {i:chosen for i,_,chosen in values}
    enabled=actual(boundaries)
    off=copy.deepcopy(boundaries)
    for x in off:x['env'].pop(plan.DEPTH8)
    disabled=actual(off)
    if enabled!=disabled or 6003 in enabled.values():
        raise ValueError('Depth8 altered a boundary fallback')
    for x in boundaries:x['expected']=enabled[x['ID']]
    paired=[]
    for x in table['cases']:
        if not x['candidate_overlay_applied']:continue
        after=copy.deepcopy(x);after['accuracy_repeats']=3
        before=copy.deepcopy(after);before['env']=copy.deepcopy(x['historical_env'])
        before['expected']=x['historical_expected']
        paired.append(dict(ID=x['ID'],before=before,after=after))
    if len(paired)!=6:raise ValueError('Expected six formal-shape A/B pairs')
    return dict(accuracy=accuracy,boundaries=boundaries,paired=paired,rounds=3,
        storage_contract='Packed dense NN, lda=K/ldb=N/ldc=N, vector-aligned device allocations; '
                         'v1 host problem does not expose device karg pointers or strides for host inspection',
        acceptance='Correctness and actual instance must pass; preserve high CV/negative timing; no production adoption')


def run(site):
    frozen=site.manifest['qualification_plan']
    journal=site.root/'qualification_results.jsonl'
    done=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    for row in done:
        if row['plan_sha256']!=site.manifest['plan_sha256']:
            raise ValueError('Mixed qualification plan')
    completed={(x['kind'],x['ID']) for x in done}
    def spec(case,kind,variant,key,attempt):
        return dict(logical_key=key,ID=case['ID'],shape={k:case[k] for k in ['M','N','K','G']},
                    purpose='qualification',variant=variant,expected=case['expected'],env=case['env'],
                    repeats=case.get('accuracy_repeats',1) if kind=='accuracy' else 1,
                    kind=kind,attempt=attempt,command=plan.command(case,kind))
    def save(kind,i,records,**fields):
        if len({x['epoch'] for x in records})!=1 or not all(x['ok'] for x in records):
            raise ValueError('Partial or mixed-epoch qualification')
        row=dict(kind=kind,ID=i,at=r.stamp(),plan_sha256=site.manifest['plan_sha256'],
                 epoch=records[0]['epoch'],keys=[x['key'] for x in records],**fields)
        r.append(journal,row);print('QUALIFICATION_PASS',kind,i,flush=True)
    for kind,cases in [('all_integer_m_accuracy',frozen['accuracy']),('boundary_accuracy',frozen['boundaries'])]:
        for x in cases:
            if (kind,x['ID']) in completed:continue
            attempt=uuid.uuid4().hex[:16]
            record=site.execute(spec(x,'accuracy','after',f'qual_{kind}_{x["ID"]}',attempt))
            save(kind,x['ID'],[record])
    for pair in frozen['paired']:
        i=pair['ID']
        if ('paired',i) in completed:continue
        attempt=uuid.uuid4().hex[:16];records=[];times={'before':[],'after':[]};cvs={'before':[],'after':[]}
        for variant in ['before','after']:
            records.append(site.execute(spec(pair[variant],'accuracy',variant,f'qual_pair_{i}_{variant}_acc',attempt)))
        for rd in range(3):
            for variant in (['before','after'] if rd%2==0 else ['after','before']):
                record=site.execute(spec(pair[variant],'perf',variant,f'qual_pair_{i}_{variant}_r{rd+1}',attempt))
                records.append(record);times[variant].append(record['result']['median_ms']);cvs[variant].append(record['result']['cv'])
        before=statistics.median(times['before']);after=statistics.median(times['after'])
        save('paired',i,records,round_ms=times,round_cv=cvs,before_ms=before,after_ms=after,
             change_pct=(after/before-1)*100,high_cv=max(cvs['before']+cvs['after'])>.05,
             round_changes_pct=[(a/b-1)*100 for a,b in zip(times['after'],times['before'])])
    r.write(site.root/'qualification_complete.json',dict(at=r.stamp(),plan_sha256=site.manifest['plan_sha256'],
        accuracy_cases=32,boundary_cases=len(frozen['boundaries']),paired_cases=6))

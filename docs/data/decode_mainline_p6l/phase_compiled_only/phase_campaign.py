"""Bounded H3 phase factors; no GPU operation on import or plan creation."""
from pathlib import Path
import argparse,copy,json,statistics,subprocess,time,uuid
import campaign,final_confirmation as final,h3_factors_campaign as prior,resource_campaign as prior_resources,plan,runtime as r
ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'phase_discovery_plan.json'
QUAL=ROOT/'phase_qualification.json'
RESULTS=ROOT/'phase_discovery_results.json'
VARIANTS={
    'compact':dict(variant=21,instance=6026),
    'compact_scope':dict(variant=24,instance=6026),
    'wide':dict(variant=22,instance=6027),
    'hybrid':dict(variant=23,instance=None),
}

def supported(c):
    m,n,k,g=(c[z] for z in ('M','N','K','G'))
    return 1<=m<=64 and 1024<=n<=4096 and n%128==0 and 2048<=k<=8192 and k%128==0 and 1<=g<=1024

def candidate(base,name,fast=1):
    other=copy.deepcopy(base);v=VARIANTS[name]
    other['env'].update({plan.PREFIX+'ENABLE_BF16_NN_DECODE_MAINLINE':'1',
        plan.PREFIX+'DISABLE_BF16_NN_DECODE_MAINLINE':'0',
        plan.PREFIX+'BF16_NN_DECODE_MAINLINE_VARIANT':str(v['variant']),
        plan.PREFIX+'ENABLE_BF16_NN_DECODE_FAST_POLICY':str(fast),
        plan.PREFIX+'DISABLE_BF16_NN_DECODE_FAST_POLICY':'0'})
    enabled=supported(base) and (name!='wide' or base['M']>=17) and (name not in ['compact_scope','hybrid'] or base['M']>=17 or base['K']>=6144)
    if enabled:other['expected']=(6026 if base['M']<=16 else 6027) if name=='hybrid' else v['instance']
    other['variant']=name
    return other

def create():
    hypothesis=r.read(ROOT/'h3_resource_factors_hypothesis.json')
    source={x['ID']:x for x in r.read(ROOT/'source_results.json')}
    pairs=[]
    def add(key,base,other,name,control=False,**fields):
        pairs.append(dict(key=key,ID=base['ID'],base=base,candidate=other,variant=name,control=control,**fields))
    for name in ['compact','wide']:
        for ident in hypothesis['discovery_representative_ids']:
            base=prior.as_case(source[ident]);other=candidate(base,name)
            if other['expected']!=base['expected']:add(f'id{ident}_{name}',base,other,name)
    for ident,name in [(358,'compact'),(622,'wide'),(1184,'compact'),(1185,'wide'),(147,'compact'),(35,'wide'),(788,'wide')]:
        base=prior.as_case(source[ident]);other=candidate(base,name)
        assert other['expected']==base['expected']
        add(f'fallback_id{ident}_{name}',base,other,name,True,fallback=True)
    for ident,name in [(621,'compact'),(805,'wide'),(1009,'compact'),(312,'wide')]:
        base=prior.as_case(source[ident])
        add(f'phase_id{ident}_{name}',prior_resources.candidate(base,name),candidate(base,name),name,True,phase_compare=True)
    for ident,name in [(1009,'compact'),(621,'wide')]:
        base=prior.as_case(source[ident]);a=candidate(base,name);b=candidate(base,'hybrid')
        assert a['expected']==b['expected']
        add(f'route_id{ident}_{name}_hybrid',a,b,'hybrid',True,route_compare=True)
    qualifications=[]
    def synthetic(m,k=2176):
        return dict(ID=-40000-len(qualifications),M=m,N=1152,K=k,G=3,phase='decode',expected=5000,env={plan.V7:'1'},accuracy_repeats=3)
    for name,ms in [('compact',[1,15,16,17,31,32,33,63,64]),('wide',[17,31,32,33,63,64])]:
        for m in ms:qualifications.append(candidate(synthetic(m),name))
    for name,m,k in [('compact_scope',1,6144),('compact_scope',17,2176),('hybrid',1,6144),('hybrid',17,2176)]:
        qualifications.append(candidate(synthetic(m,k),name))
    for name in ['compact','wide']:
        case=candidate(synthetic(64),name);case['env'][plan.PREFIX+'DISABLE_BF16_NN_DECODE_FAST_POLICY']='1'
        case['variant']+='_fast_disabled';qualifications.append(case)
    for ident,name in [(620,'compact'),(805,'wide')]:
        base=prior.as_case(source[ident]);case=candidate(base,name)
        case['ID']=-40000-len(qualifications);case['env'][plan.PREFIX+'DISABLE_BF16_NN_DECODE_MAINLINE']='1'
        case['expected']=base['expected'];case['variant']+='_family_disabled';qualifications.append(case)
    for name in ['compact_scope','hybrid']:
        base=synthetic(1)
        # N1152 is not a V7 N256 tile multiple; the real disabled selector
        # returns legacy4002 here, verified by the CPU policy probe.
        base['expected']=4002
        case=candidate(base,name);assert case['expected']==4002
        case['variant']+='_scope_fallback';qualifications.append(case)
    assert len(pairs)==46 and len(qualifications)==25
    result=dict(schema=1,hypothesis_sha256=plan.sha(ROOT/'phase_hypothesis.json'),
        variants=VARIANTS,qualification_cases=qualifications,discovery_pairs=pairs,
        protocol=r.read(ROOT/'research_plan.json')['diagnostic_protocol'],
        selection='Predeclared19 representatives; all actual compact/wide candidates,7 fallback,4 phase and2 identical-GPU route controls. No ID-based adoption.',
        CPU_only_creation=True,no_new_heldout_draw=True,no_full_GPU_run=True)
    result['sha256']=plan.digest(result)
    return result

def prepare():
    proposed=create()
    if PLAN.exists():assert r.read(PLAN)==proposed
    else:r.write(PLAN,proposed)
    cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
    audit=prior.selector_check(cases,ROOT/'cpu_phase_campaign_preparation')
    r.write(ROOT/'phase_campaign_cpu_selector_audit.json',audit)
    predict,command=final.predictor(ROOT/'cpu_phase_campaign_baseline1193')
    originals=[prior.as_case(x) for x in r.read(ROOT/'source_results.json')]
    assert len(originals)==1193 and predict(originals)==[x['expected'] for x in originals]
    r.write(ROOT/'phase_baseline1193_cpu_audit.json',dict(ok=True,cpu_only=True,GPU=False,cases=1193,command=command,selector_sha256=plan.sha(plan.SELECTOR)))
    print('PHASE_PLAN_CPU_PASS',len(cases),'cases;46 pairs;27 qualification jobs; no GPU',flush=True)

def freeze():
    assert time.time()+60<r.read(ROOT/'session.json')['gpu_deadline_epoch']
    folder=ROOT/'compile/build9';stages=r.read(folder/'results.json')
    assert len(stages)==2 and all(x['rc']==0 for x in stages)
    actual=r.read(ROOT/'phase_build_isa_audit.json');assert actual['ok'] and actual['old_shader_count']==1710
    assert actual['benchmark_sha256']==plan.sha(plan.BENCH)
    policy=r.read(ROOT/'cpu_phase_policy/results.json')
    assert policy['ok'] and policy['case_count']==89493
    assert policy['new_source_sha256']==plan.sha(plan.REPO/'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp')
    assert r.read(ROOT/'cpu_phase_policy/public_disable_equivalence.json')['ok']
    assert r.read(ROOT/'phase_baseline1193_cpu_audit.json')['ok']
    proposed=create();assert r.read(PLAN)==proposed
    cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
    for case in cases:plan.environment(case['env'],{})
    audit=prior.selector_check(cases,ROOT/'cpu_phase_campaign')
    r.write(ROOT/'phase_campaign_cpu_selector_audit.json',audit)
    source=r.read(folder/'source_sha256.json')
    assert all(plan.sha(plan.REPO/n)==sha for n,sha in source.items())
    old=r.read(ROOT/'frozen_plan.json');m=copy.deepcopy(old);m.pop('plan_sha256',None)
    m.update(phase='h3_phase_discovery',candidate_plan_sha256=proposed['sha256'],parent_manifest_sha256=old['plan_sha256'],benchmark_sha256=plan.sha(plan.BENCH),selector_sha256=plan.sha(plan.SELECTOR))
    for key in ['final_plan_sha256','regression_profile_plan_sha256','resource_profile_plan_sha256']:m.pop(key,None)
    deps=plan.base.dependencies();m['dependencies']['sha256'].update(deps['sha256']);m['dependencies']['ldd']=deps['ldd']
    for name in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
        p=plan.REPO/'build/bin'/name;m['dependencies']['sha256'][str(p)]=plan.sha(p)
    m['worktree_source_sha256'].update(source)
    for name in m['worktree_source_sha256']:m['worktree_source_sha256'][name]=plan.sha(plan.REPO/name)
    new=['phase_campaign.py','phase_discovery_plan.json','phase_hypothesis.json','phase_build_isa_audit.json','audit_phase_build.py','phase_profiles_campaign.py','phase_profile_rule.json','build_phase_factor.py','audit_phase_qualification.py','analyze_phase_discovery.py','run_phase_discovery.py','run_final_confirmation_phase.py','cpu_phase_policy_driver.py','validate_phase_public_masks.py','cpu_phase_policy/results.json','cpu_phase_policy/public_disable_equivalence.json','cpu_phase_policy/heldout_geometry_inventory.json','phase_baseline1193_cpu_audit.json','drafts/resource_interleave/audit_phase_probe.py']
    for name in list(m['runner_sha256'])+new:m['runner_sha256'][name]=plan.sha(ROOT/name)
    m['plan_sha256']=plan.digest(m);r.write(ROOT/'manifests'/(m['plan_sha256']+'.json'),m);r.write(ROOT/'frozen_plan.json',m)
    print('PHASE_GPU_PLAN_FROZEN',m['plan_sha256'],flush=True)

def qualify(site):
    proposed=r.read(PLAN);done=r.read(QUAL,[])
    for name in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
        key='phase_regression_'+name
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        binary=plan.REPO/'build/bin'/name
        spec=dict(kind='regression',logical_key=key,command=[str(binary)],binary=str(binary),expected_binary_sha256=plan.sha(binary),env={},regression_kind='device_args_query_test' if name.endswith('query_test') else 'device_args_smoke',purpose='phase_qualification')
        done.append(site.execute(spec));r.write(QUAL,done)
    for case in proposed['qualification_cases']:
        key=f'phase_qual_{case["ID"]}_{case["variant"]}'
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        spec=campaign.ck(case,'accuracy',key);spec.update(variant=case['variant'],purpose='phase_qualification')
        done.append(site.execute(spec));r.write(QUAL,done)
    assert len([x for x in done if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch()])==27
    print('PHASE_QUALIFICATION_PASS27',flush=True)

def discover(site):
    qualified=r.read(QUAL,[])
    assert len([x for x in qualified if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() and x['ok']])==27
    completed=r.read(RESULTS,[])
    for pair in r.read(PLAN)['discovery_pairs']:
        if any(x['pair']==pair['key'] and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in completed):continue
        attempt=uuid.uuid4().hex[:12];epoch=site.epoch();jobs=[]
        for arm in ['base','candidate']:
            s=campaign.ck(pair[arm],'accuracy',f'phase_{pair["key"]}_{attempt}_{arm}_accuracy')
            jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],purpose='phase_discovery')))
        for rd in [1,2,3]:
            for arm in (['base','candidate'] if rd%2 else ['candidate','base']):
                s=campaign.ck(pair[arm],'perf',f'phase_{pair["key"]}_{attempt}_{arm}_r{rd}')
                jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],round=rd,purpose='phase_discovery')))
        assert len(jobs)==8 and all(x['ok'] and x['epoch']==epoch for x in jobs)
        aa=[x['result'] for x in jobs if x['kind']=='perf' and x['variant']=='base']
        bb=[x['result'] for x in jobs if x['kind']=='perf' and x['variant']=='candidate']
        a=statistics.median(x['median_ms'] for x in aa);b=statistics.median(x['median_ms'] for x in bb)
        row=dict(pair=pair['key'],ID=pair['ID'],shape={z:pair['base'][z] for z in ['M','N','K','G']},
            variant=pair['variant'],control=pair['control'],base_instance=pair['base']['expected'],candidate_instance=pair['candidate']['expected'],
            CK_base_ms=a,CK_candidate_ms=b,base_ms=a,candidate_ms=b,frozen_BLAS_ms=pair['base'].get('frozen_BLAS_ms'),
            route_compare=pair.get('route_compare',False),phase_compare=pair.get('phase_compare',False),fallback=pair.get('fallback',False),
            ratio_candidate_over_base=b/a,delta_pct=100*(b/a-1),
            high_cv=any(x['cv']>.05 for x in aa+bb),base_rounds=aa,candidate_rounds=bb,
            epoch=epoch,plan_sha256=site.manifest['plan_sha256'],binary_sha256=site.manifest['benchmark_sha256'],keys=[x['key'] for x in jobs],
            accuracy_pass=True,not_final_confirmation=True)
        completed.append(row);r.write(RESULTS,completed)
        print('PHASE_DISCOVERY_PAIR_PASS',pair['key'],'delta_pct',round(row['delta_pct'],3),'highCV',row['high_cv'],flush=True)
    assert len([x for x in completed if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch()])==46
    print('PHASE_DISCOVERY_COMPLETE46',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','freeze','qualify','discover']);args=ap.parse_args();r.no_core()
    if args.stage=='prepare':prepare();return
    with r.locked():
        if args.stage=='freeze':freeze()
        else:
            site=r.Site(ROOT);assert site.manifest['phase']=='h3_phase_discovery'
            qualify(site) if args.stage=='qualify' else discover(site)

if __name__=='__main__':main()

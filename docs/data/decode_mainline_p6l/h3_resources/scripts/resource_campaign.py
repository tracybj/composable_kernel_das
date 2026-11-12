"""Bounded H3 resource factors; no GPU operation on import or plan creation."""
from pathlib import Path
import argparse,copy,json,statistics,subprocess,time,uuid
import campaign,final_confirmation as final,h3_factors_campaign as prior,plan,runtime as r
ROOT=Path(__file__).resolve().parent
ISO=ROOT/'isolated_resource_build'
PLAN=ROOT/'resource_discovery_plan.json'
QUAL=ROOT/'resource_qualification.json'
RESULTS=ROOT/'resource_discovery_results.json'
VARIANTS={
    'compact':dict(variant=17,instance=6024),
    'compact_scope':dict(variant=18,instance=6024),
    'wide':dict(variant=19,instance=6025),
    'hybrid':dict(variant=20,instance=None),
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
    enabled=supported(base) and (name!='wide' or base['M']>=17) and (name!='compact_scope' or base['M']>=17 or base['K']>=6144)
    if enabled:other['expected']=(6024 if base['M']<=16 else 6025) if name=='hybrid' else v['instance']
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
        add(f'resource_id{ident}_{name}',prior.candidate(base,'coop_d4_fast'),candidate(base,name),name,True,resource_compare=True)
    for ident,name in [(1009,'compact'),(621,'wide')]:
        base=prior.as_case(source[ident]);a=candidate(base,name);b=candidate(base,'hybrid')
        assert a['expected']==b['expected']
        add(f'route_id{ident}_{name}_hybrid',a,b,'hybrid',True,route_compare=True)
    qualifications=[]
    def synthetic(m,k=2176):
        return dict(ID=-30000-len(qualifications),M=m,N=1152,K=k,G=3,phase='decode',expected=5000,env={plan.V7:'1'},accuracy_repeats=3)
    for name,ms in [('compact',[1,15,16,17,31,32,33,63,64]),('wide',[17,31,32,33,63,64])]:
        for m in ms:qualifications.append(candidate(synthetic(m),name))
    for name,m,k in [('compact_scope',1,6144),('compact_scope',17,2176),('hybrid',1,2176),('hybrid',17,2176)]:
        qualifications.append(candidate(synthetic(m,k),name))
    for name in ['compact','wide']:
        case=candidate(synthetic(64),name);case['env'][plan.PREFIX+'DISABLE_BF16_NN_DECODE_FAST_POLICY']='1'
        case['variant']+='_fast_disabled';qualifications.append(case)
    for ident,name in [(620,'compact'),(805,'wide')]:
        base=prior.as_case(source[ident]);case=candidate(base,name)
        case['ID']=-30000-len(qualifications);case['env'][plan.PREFIX+'DISABLE_BF16_NN_DECODE_MAINLINE']='1'
        case['expected']=base['expected'];case['variant']+='_family_disabled';qualifications.append(case)
    assert len(pairs)==46 and len(qualifications)==23
    result=dict(schema=1,hypothesis_sha256=plan.sha(ROOT/'h3_resource_factors_hypothesis.json'),
        variants=VARIANTS,qualification_cases=qualifications,discovery_pairs=pairs,
        protocol=r.read(ROOT/'research_plan.json')['diagnostic_protocol'],
        selection='Predeclared19 representatives; all actual compact/wide candidates,7 fallback,4 resource and2 identical-GPU route controls. No ID-based adoption.',
        CPU_only_creation=True,no_new_heldout_draw=True,no_full_GPU_run=True)
    result['sha256']=plan.digest(result)
    return result

def prepare():
    proposed=create()
    if PLAN.exists():assert r.read(PLAN)==proposed
    else:r.write(PLAN,proposed)
    cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
    original_repo,original_selector=plan.REPO,plan.SELECTOR
    try:
        plan.REPO=ISO/'source';plan.SELECTOR=plan.REPO/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'
        audit=prior.selector_check(cases,ISO/'cpu_resource_campaign')
    finally:plan.REPO=original_repo;plan.SELECTOR=original_selector
    r.write(ISO/'resource_campaign_cpu_selector_audit.json',audit)
    print('RESOURCE_PLAN_CPU_PASS',len(cases),'cases;46 pairs;25 qualification jobs; no GPU',flush=True)

def freeze():
    assert time.time()+60<r.read(ROOT/'session.json')['gpu_deadline_epoch']
    assert r.read(ROOT/'profile_delivery_batch7.json')['captures']==8
    promotion=r.read(ROOT/'resource_promotion.json');assert promotion['complete']
    assert promotion['new_benchmark_sha256']==plan.sha(plan.BENCH)
    proposed=create();assert r.read(PLAN)==proposed
    cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
    for case in cases:plan.environment(case['env'],{})
    audit=prior.selector_check(cases,ROOT/'cpu_resource_campaign')
    r.write(ROOT/'resource_campaign_cpu_selector_audit.json',audit)
    for name,sha in promotion['source_sha256'].items():assert plan.sha(plan.REPO/name)==sha
    old=r.read(ROOT/'frozen_plan.json');m=copy.deepcopy(old);m.pop('plan_sha256',None)
    m.update(phase='h3_resource_discovery',candidate_plan_sha256=proposed['sha256'],
        benchmark_sha256=plan.sha(plan.BENCH),selector_sha256=plan.sha(plan.SELECTOR))
    deps=plan.base.dependencies();m['dependencies']['sha256'].update(deps['sha256']);m['dependencies']['ldd']=deps['ldd']
    for name in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
        p=plan.REPO/'build/bin'/name;m['dependencies']['sha256'][str(p)]=plan.sha(p)
    m['worktree_source_sha256'].update(promotion['source_sha256'])
    for name in m['worktree_source_sha256']:m['worktree_source_sha256'][name]=plan.sha(plan.REPO/name)
    for name in list(m['runner_sha256'])+['resource_campaign.py','resource_discovery_plan.json','resource_promotion.json','resource_mechanism_decision.json','h3_resource_factors_hypothesis.json','audit_resource_isa.py','resource_profiles_campaign.py','resource_profile_rule.json','promote_resource_build.py','audit_resource_qualification.py','analyze_resource_discovery.py','run_resource_discovery.py','run_final_confirmation_resources.py']:
        m['runner_sha256'][name]=plan.sha(ROOT/name)
    m['plan_sha256']=plan.digest(m);r.write(ROOT/'manifests'/(m['plan_sha256']+'.json'),m);r.write(ROOT/'frozen_plan.json',m)
    print('RESOURCE_GPU_PLAN_FROZEN',m['plan_sha256'],flush=True)

def qualify(site):
    proposed=r.read(PLAN);done=r.read(QUAL,[])
    for name in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
        key='resource_regression_'+name
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        binary=plan.REPO/'build/bin'/name
        spec=dict(kind='regression',logical_key=key,command=[str(binary)],binary=str(binary),expected_binary_sha256=plan.sha(binary),env={},regression_kind='device_args_query_test' if name.endswith('query_test') else 'device_args_smoke',purpose='resource_qualification')
        done.append(site.execute(spec));r.write(QUAL,done)
    for case in proposed['qualification_cases']:
        key=f'resource_qual_{case["ID"]}_{case["variant"]}'
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        spec=campaign.ck(case,'accuracy',key);spec.update(variant=case['variant'],purpose='resource_qualification')
        done.append(site.execute(spec));r.write(QUAL,done)
    assert len([x for x in done if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch()])==25
    print('RESOURCE_QUALIFICATION_PASS25',flush=True)

def discover(site):
    qualified=r.read(QUAL,[])
    assert len([x for x in qualified if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() and x['ok']])==25
    completed=r.read(RESULTS,[])
    for pair in r.read(PLAN)['discovery_pairs']:
        if any(x['pair']==pair['key'] and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in completed):continue
        attempt=uuid.uuid4().hex[:12];epoch=site.epoch();jobs=[]
        for arm in ['base','candidate']:
            s=campaign.ck(pair[arm],'accuracy',f'resource_{pair["key"]}_{attempt}_{arm}_accuracy')
            jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],purpose='resource_discovery')))
        for rd in [1,2,3]:
            for arm in (['base','candidate'] if rd%2 else ['candidate','base']):
                s=campaign.ck(pair[arm],'perf',f'resource_{pair["key"]}_{attempt}_{arm}_r{rd}')
                jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],round=rd,purpose='resource_discovery')))
        assert len(jobs)==8 and all(x['ok'] and x['epoch']==epoch for x in jobs)
        aa=[x['result'] for x in jobs if x['kind']=='perf' and x['variant']=='base']
        bb=[x['result'] for x in jobs if x['kind']=='perf' and x['variant']=='candidate']
        a=statistics.median(x['median_ms'] for x in aa);b=statistics.median(x['median_ms'] for x in bb)
        row=dict(pair=pair['key'],ID=pair['ID'],shape={z:pair['base'][z] for z in ['M','N','K','G']},
            variant=pair['variant'],control=pair['control'],base_instance=pair['base']['expected'],candidate_instance=pair['candidate']['expected'],
            CK_base_ms=a,CK_candidate_ms=b,base_ms=a,candidate_ms=b,frozen_BLAS_ms=pair['base'].get('frozen_BLAS_ms'),
            route_compare=pair.get('route_compare',False),resource_compare=pair.get('resource_compare',False),fallback=pair.get('fallback',False),
            ratio_candidate_over_base=b/a,delta_pct=100*(b/a-1),
            high_cv=any(x['cv']>.05 for x in aa+bb),base_rounds=aa,candidate_rounds=bb,
            epoch=epoch,plan_sha256=site.manifest['plan_sha256'],binary_sha256=site.manifest['benchmark_sha256'],keys=[x['key'] for x in jobs],
            accuracy_pass=True,not_final_confirmation=True)
        completed.append(row);r.write(RESULTS,completed)
        print('RESOURCE_DISCOVERY_PAIR_PASS',pair['key'],'delta_pct',round(row['delta_pct'],3),'highCV',row['high_cv'],flush=True)
    assert len([x for x in completed if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch()])==46
    print('RESOURCE_DISCOVERY_COMPLETE46',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','freeze','qualify','discover']);args=ap.parse_args();r.no_core()
    if args.stage=='prepare':prepare();return
    with r.locked():
        if args.stage=='freeze':freeze()
        else:
            site=r.Site(ROOT);assert site.manifest['phase']=='h3_resource_discovery'
            qualify(site) if args.stage=='qualify' else discover(site)

if __name__=='__main__':main()

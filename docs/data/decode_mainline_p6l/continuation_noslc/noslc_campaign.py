"""Bounded H3 phase factors; no GPU operation on import or plan creation."""
from pathlib import Path
import argparse,copy,json,statistics,subprocess,time,uuid
import campaign,final_confirmation as final,h3_factors_campaign as prior,resource_campaign as prior_resources,plan,runtime as r
import low_lds_campaign as parent
ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'noslc_discovery_plan.json'
QUAL=ROOT/'noslc_qualification.json'
RESULTS=ROOT/'noslc_discovery_results.json'
VARIANTS={'noslc_full':dict(variant=41,instance=6035),'noslc':dict(variant=42,instance=6035)}

def supported(c):
 m,n,k,g=(c[z] for z in ('M','N','K','G'))
 return 1<=m<=64 and 1024<=n<=4096 and n%128==0 and 2048<=k<=8192 and k%128==0 and 1<=g<=1024

def candidate(base,name,fast=1):
 other=copy.deepcopy(base);v=VARIANTS[name]
 other['env'].update({plan.PREFIX+'ENABLE_BF16_NN_DECODE_MAINLINE':'1',plan.PREFIX+'DISABLE_BF16_NN_DECODE_MAINLINE':'0',plan.PREFIX+'BF16_NN_DECODE_MAINLINE_VARIANT':str(v['variant']),plan.PREFIX+'ENABLE_BF16_NN_DECODE_FAST_POLICY':str(fast),plan.PREFIX+'DISABLE_BF16_NN_DECODE_FAST_POLICY':'0'})
 if supported(base) and (name=='noslc_full' or base['M']>=17):other['expected']=6035
 other['variant']=name
 return other

def create():
 import wave4_campaign as registered_parent
 old=registered_parent.create();data=copy.deepcopy(old)
 data['discovery_pairs']=[x for x in data['discovery_pairs'] if not x.get('geometry_compare')]
 assert len(data['discovery_pairs'])==23
 for pair in data['discovery_pairs']:
  pair['key']=pair['key'].replace('scope_wave4','noslc');pair['variant']='noslc'
  pair['candidate']['variant']='noslc'
 for case in data['qualification_cases']:case['variant']=case['variant'].replace('scope_wave4','noslc').replace('wave4','noslc_full')
 data.pop('sha256',None);data.update(hypothesis_sha256=plan.sha(ROOT/'noslc_mechanism_decision.json'),variants=VARIANTS,selection='Sameall16registeredM17..64representativesand7fallbacks. Cacheonlyrevisionof6035; same96originals/53weak. Noold/newbinarytimingmixture; noheldoutdraw. Directold6035cachecomparisonisstaticISAonly; formalbaselineP6kbestretestedinsidefreshbinary.')
 data['sha256']=plan.digest(data);return data

def prepare():
    proposed=create()
    if PLAN.exists():assert r.read(PLAN)==proposed
    else:r.write(PLAN,proposed)
    cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
    audit=prior.selector_check(cases,ROOT/'cpu_noslc_campaign_preparation')
    r.write(ROOT/'noslc_campaign_cpu_selector_audit.json',audit)
    predict,command=final.predictor(ROOT/'cpu_noslc_campaign_baseline1193')
    originals=[prior.as_case(x) for x in r.read(ROOT/'source_results.json')]
    assert len(originals)==1193 and predict(originals)==[x['expected'] for x in originals]
    r.write(ROOT/'noslc_baseline1193_cpu_audit.json',dict(ok=True,cpu_only=True,GPU=False,cases=1193,command=command,selector_sha256=plan.sha(plan.SELECTOR)))
    print('PHASE_PLAN_CPU_PASS',len(cases),'cases;23 pairs;35 qualification jobs; no GPU',flush=True)

def freeze():
    assert time.time()+60<r.read(ROOT/'session.json')['gpu_deadline_epoch']
    folder=ROOT/'compile/build15';stages=r.read(folder/'results.json')
    assert len(stages)==1 and all(x['rc']==0 for x in stages)
    actual=r.read(ROOT/'noslc_build_isa_audit.json');assert actual['ok'] and actual['old_shader_count']==1766
    assert actual['benchmark_sha256']==plan.sha(plan.BENCH)
    policy=r.read(ROOT/'drafts/conditional_n128_wave4/cpu_policy/results.json')
    assert policy['ok'] and policy['case_count']==175389
    assert policy['new_source_sha256']==plan.sha(plan.REPO/'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp')
    assert r.read(ROOT/'drafts/conditional_n128_wave4/cpu_policy/public_disable_equivalence.json')['ok']
    assert r.read(ROOT/'noslc_baseline1193_cpu_audit.json')['ok']
    proposed=create();assert r.read(PLAN)==proposed
    cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
    for case in cases:plan.environment(case['env'],{})
    audit=prior.selector_check(cases,ROOT/'cpu_noslc_campaign')
    r.write(ROOT/'noslc_campaign_cpu_selector_audit.json',audit)
    source=r.read(folder/'source_sha256.json')
    assert all(plan.sha(plan.REPO/n)==sha for n,sha in source.items())
    old=r.read(ROOT/'frozen_plan.json');m=copy.deepcopy(old);m.pop('plan_sha256',None)
    m.update(phase='h3_noslc_discovery',candidate_plan_sha256=proposed['sha256'],parent_manifest_sha256=old['plan_sha256'],benchmark_sha256=plan.sha(plan.BENCH),selector_sha256=plan.sha(plan.SELECTOR))
    for key in ['final_plan_sha256','regression_profile_plan_sha256','resource_profile_plan_sha256','phase_profile_plan_sha256','k64_profile_plan_sha256','n256_profile_plan_sha256','b_reuse_profile_plan_sha256','low_lds_profile_plan_sha256','wave4_profile_plan_sha256','noslc_profile_plan_sha256']:m.pop(key,None)
    deps=plan.base.dependencies();m['dependencies']['sha256'].update(deps['sha256']);m['dependencies']['ldd']=deps['ldd']
    for name in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
        p=plan.REPO/'build/bin'/name;m['dependencies']['sha256'][str(p)]=plan.sha(p)
    m['worktree_source_sha256'].update(source)
    for name in m['worktree_source_sha256']:m['worktree_source_sha256'][name]=plan.sha(plan.REPO/name)
    new=['noslc_final_selection_rule.json','noslc_campaign.py','noslc_discovery_plan.json','noslc_mechanism_decision.json','noslc_build_isa_audit.json','audit_noslc_build.py','noslc_profiles_campaign.py','noslc_profile_rule.json','audit_noslc_qualification.py','analyze_noslc_discovery.py','run_noslc_discovery.py','noslc_baseline1193_cpu_audit.json','drafts/conditional_n128_wave4/build_factor.py','drafts/conditional_n128_wave4/audit_schedule.py','drafts/conditional_n128_wave4/cpu_policy/results.json','drafts/conditional_n128_wave4/cpu_policy/public_disable_equivalence.json','drafts/conditional_n128_wave4/heldout_geometry_inventory.json']
    new += ['drafts/conditional_n128_wave4/conditional_campaign_plan.json','drafts/conditional_n128_wave4/mechanism_decision.json','drafts/conditional_n128_wave4/old_default_compatibility.json','drafts/conditional_n128_wave4/cpu_model_result.json','drafts/conditional_n128_wave4/probe_resources.json','drafts/conditional_n128_wave4/actual_cfg_proof.json']
    new += ['drafts/conditional_A_cache_hint/revision_n128_wave4/cache_only_isa_proof.json','drafts/conditional_A_cache_hint/revision_n128_wave4/build_revision.py']
    new+= [str(p.relative_to(ROOT)) for p in ROOT.glob('*.py')]
    for name in list(m['runner_sha256'])+new:m['runner_sha256'][name]=plan.sha(ROOT/name)
    m['plan_sha256']=plan.digest(m);r.write(ROOT/'manifests'/(m['plan_sha256']+'.json'),m);r.write(ROOT/'frozen_plan.json',m)
    print('PHASE_GPU_PLAN_FROZEN',m['plan_sha256'],flush=True)

def qualify(site):
    proposed=r.read(PLAN);done=r.read(QUAL,[])
    for name in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
        key='noslc_regression_'+name
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        binary=plan.REPO/'build/bin'/name
        spec=dict(kind='regression',logical_key=key,command=[str(binary)],binary=str(binary),expected_binary_sha256=plan.sha(binary),env={},regression_kind='device_args_query_test' if name.endswith('query_test') else 'device_args_smoke',purpose='noslc_qualification')
        done.append(site.execute(spec));r.write(QUAL,done)
    for case in proposed['qualification_cases']:
        key=f'noslc_qual_{case["ID"]}_{case["variant"]}'
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        spec=campaign.ck(case,'accuracy',key);spec.update(variant=case['variant'],purpose='noslc_qualification')
        done.append(site.execute(spec));r.write(QUAL,done)
    assert len([x for x in done if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch()])==35
    print('WAVE4_QUALIFICATION_PASS35',flush=True)

def discover(site):
    qualified=r.read(QUAL,[])
    assert len([x for x in qualified if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() and x['ok']])==35
    completed=r.read(RESULTS,[])
    for pair in r.read(PLAN)['discovery_pairs']:
        if any(x['pair']==pair['key'] and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in completed):continue
        attempt=uuid.uuid4().hex[:12];epoch=site.epoch();jobs=[]
        for arm in ['base','candidate']:
            s=campaign.ck(pair[arm],'accuracy',f'noslc_{pair["key"]}_{attempt}_{arm}_accuracy')
            jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],purpose='noslc_discovery')))
        for rd in [1,2,3]:
            for arm in (['base','candidate'] if rd%2 else ['candidate','base']):
                s=campaign.ck(pair[arm],'perf',f'noslc_{pair["key"]}_{attempt}_{arm}_r{rd}')
                jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],round=rd,purpose='noslc_discovery')))
        assert len(jobs)==8 and all(x['ok'] and x['epoch']==epoch for x in jobs)
        aa=[x['result'] for x in jobs if x['kind']=='perf' and x['variant']=='base']
        bb=[x['result'] for x in jobs if x['kind']=='perf' and x['variant']=='candidate']
        a=statistics.median(x['median_ms'] for x in aa);b=statistics.median(x['median_ms'] for x in bb)
        row=dict(pair=pair['key'],ID=pair['ID'],shape={z:pair['base'][z] for z in ['M','N','K','G']},
            variant=pair['variant'],control=pair['control'],base_instance=pair['base']['expected'],candidate_instance=pair['candidate']['expected'],
            CK_base_ms=a,CK_candidate_ms=b,base_ms=a,candidate_ms=b,frozen_BLAS_ms=pair['base'].get('frozen_BLAS_ms'),
            route_compare=pair.get('route_compare',False),geometry_compare=pair.get('geometry_compare',False),depth_compare=pair.get('depth_compare',False),fallback=pair.get('fallback',False),
            ratio_candidate_over_base=b/a,delta_pct=100*(b/a-1),
            high_cv=any(x['cv']>.05 for x in aa+bb),base_rounds=aa,candidate_rounds=bb,
            epoch=epoch,plan_sha256=site.manifest['plan_sha256'],binary_sha256=site.manifest['benchmark_sha256'],keys=[x['key'] for x in jobs],
            accuracy_pass=True,not_final_confirmation=True)
        completed.append(row);r.write(RESULTS,completed)
        print('PHASE_DISCOVERY_PAIR_PASS',pair['key'],'delta_pct',round(row['delta_pct'],3),'highCV',row['high_cv'],flush=True)
    assert len([x for x in completed if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch()])==23
    print('WAVE4_DISCOVERY_COMPLETE23',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','freeze','qualify','discover']);args=ap.parse_args();r.no_core()
    if args.stage=='prepare':prepare();return
    with r.locked():
        if args.stage=='freeze':freeze()
        else:
            site=r.Site(ROOT);assert site.manifest['phase']=='h3_noslc_discovery'
            qualify(site) if args.stage=='qualify' else discover(site)

if __name__=='__main__':main()

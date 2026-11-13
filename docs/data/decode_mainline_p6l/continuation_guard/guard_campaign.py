"""Bounded H3 phase factors; no GPU operation on import or plan creation."""
from pathlib import Path
import argparse,copy,json,statistics,subprocess,time,uuid
import campaign,final_confirmation as final,h3_factors_campaign as prior,resource_campaign as prior_resources,plan,runtime as r
import low_lds_campaign as parent
ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'guard_discovery_plan.json'
QUAL=ROOT/'guard_qualification.json'
RESULTS=ROOT/'guard_discovery_results.json'
VARIANTS={'guard_full':dict(variant=41,instance=6035),'guard':dict(variant=42,instance=6035)}

def supported(c):
 m,n,k,g=(c[z] for z in ('M','N','K','G'))
 return 1<=m<=64 and 1024<=n<=4096 and n%128==0 and 2048<=k<=8192 and k%128==0 and 1<=g<=1024

def candidate(base,name,fast=1):
 other=copy.deepcopy(base);v=VARIANTS[name]
 other['env'].update({plan.PREFIX+'ENABLE_BF16_NN_DECODE_MAINLINE':'1',plan.PREFIX+'DISABLE_BF16_NN_DECODE_MAINLINE':'0',plan.PREFIX+'BF16_NN_DECODE_MAINLINE_VARIANT':str(v['variant']),plan.PREFIX+'ENABLE_BF16_NN_DECODE_FAST_POLICY':str(fast),plan.PREFIX+'DISABLE_BF16_NN_DECODE_FAST_POLICY':'0'})
 if supported(base) and (name=='guard_full' or base['M']>=17):other['expected']=6035
 other['variant']=name
 return other

def create():
 data=r.read(ROOT/'drafts/policy_fallback_guard/plan.json')
 data['sha256']=plan.digest(data)
 return data

def prepare():
    proposed=create()
    if PLAN.exists():assert r.read(PLAN)==proposed
    else:r.write(PLAN,proposed)
    cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
    audit=prior.selector_check(cases,ROOT/'cpu_guard_campaign_preparation')
    r.write(ROOT/'guard_campaign_cpu_selector_audit.json',audit)
    predict,command=final.predictor(ROOT/'cpu_guard_campaign_baseline1193')
    originals=[prior.as_case(x) for x in r.read(ROOT/'source_results.json')]
    assert len(originals)==1193 and predict(originals)==[x['expected'] for x in originals]
    r.write(ROOT/'guard_baseline1193_cpu_audit.json',dict(ok=True,cpu_only=True,GPU=False,cases=1193,command=command,selector_sha256=plan.sha(plan.SELECTOR)))
    print('PHASE_PLAN_CPU_PASS',len(cases),'cases;10 pairs;35 qualification jobs; no GPU',flush=True)

def freeze():
 assert time.time()+60<r.read(ROOT/'session.json')['gpu_deadline_epoch']
 folder=ROOT/'compile/build16';audit=r.read(ROOT/'guard_build_isa_audit.json')
 assert audit['ok'] and audit['all_shader_inventory_and_bytes_unchanged']==1766
 assert audit['benchmark_sha256']==plan.sha(plan.BENCH)
 proof=r.read(ROOT/'drafts/policy_fallback_guard/policy_equivalence.json')
 assert proof['ok'] and proof['cases']==322845
 assert proof['source_sha256']==plan.sha(plan.REPO/'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp')
 assert r.read(ROOT/'guard_baseline1193_cpu_audit.json')['ok']
 proposed=create();assert r.read(PLAN)==proposed
 cases=proposed['qualification_cases']+[c for p in proposed['discovery_pairs'] for c in [p['base'],p['candidate']]]
 for case in cases:plan.environment(case['env'],{})
 check=prior.selector_check(cases,ROOT/'cpu_guard_campaign')
 r.write(ROOT/'guard_campaign_cpu_selector_audit.json',check)
 sources=r.read(folder/'source_sha256.json');assert len(sources)==26
 assert all(plan.sha(plan.REPO/n)==h for n,h in sources.items())
 old=r.read(ROOT/'frozen_plan.json');m=copy.deepcopy(old);m.pop('plan_sha256',None)
 m.update(phase='host_fallback_guard',candidate_plan_sha256=proposed['sha256'],parent_manifest_sha256=old['plan_sha256'],benchmark_sha256=plan.sha(plan.BENCH),selector_sha256=plan.sha(plan.SELECTOR))
 for key in list(m):
  if key.endswith('_profile_plan_sha256') or key=='final_plan_sha256':m.pop(key)
 deps=plan.base.dependencies();m['dependencies']['sha256'].update(deps['sha256']);m['dependencies']['ldd']=deps['ldd']
 for n in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
  p=plan.REPO/'build/bin'/n;m['dependencies']['sha256'][str(p)]=plan.sha(p)
 m['worktree_source_sha256']=sources
 names=[str(p.relative_to(ROOT)) for p in ROOT.glob('*.py')]
 names+=['guard_discovery_plan.json','guard_build_isa_audit.json','guard_baseline1193_cpu_audit.json','drafts/policy_fallback_guard/plan.json','drafts/policy_fallback_guard/policy_equivalence.json','drafts/policy_fallback_guard/validate.py','drafts/policy_fallback_guard/build.py']
 for n in list(m['runner_sha256'])+names:m['runner_sha256'][n]=plan.sha(ROOT/n)
 m['plan_sha256']=plan.digest(m);r.write(ROOT/'manifests'/(m['plan_sha256']+'.json'),m);r.write(ROOT/'frozen_plan.json',m)
 print('HOST_GUARD_FROZEN',m['plan_sha256'],flush=True)

def qualify(site):
    proposed=r.read(PLAN);done=r.read(QUAL,[])
    for name in ['tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke']:
        key='guard_regression_'+name
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        binary=plan.REPO/'build/bin'/name
        spec=dict(kind='regression',logical_key=key,command=[str(binary)],binary=str(binary),expected_binary_sha256=plan.sha(binary),env={},regression_kind='device_args_query_test' if name.endswith('query_test') else 'device_args_smoke',purpose='guard_qualification')
        done.append(site.execute(spec));r.write(QUAL,done)
    for case in proposed['qualification_cases']:
        key=f'guard_qual_{case["ID"]}_{case["variant"]}'
        if any(x['logical_key']==key and x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch() for x in done):continue
        spec=campaign.ck(case,'accuracy',key);spec.update(variant=case['variant'],purpose='guard_qualification')
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
            s=campaign.ck(pair[arm],'accuracy',f'guard_{pair["key"]}_{attempt}_{arm}_accuracy')
            jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],purpose='guard_discovery')))
        for rd in [1,2,3]:
            for arm in (['base','candidate'] if rd%2 else ['candidate','base']):
                s=campaign.ck(pair[arm],'perf',f'guard_{pair["key"]}_{attempt}_{arm}_r{rd}')
                jobs.append(site.execute(dict(s,variant=arm,attempt=attempt,pair=pair['key'],round=rd,purpose='guard_discovery')))
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
    assert len([x for x in completed if x['plan_sha256']==site.manifest['plan_sha256'] and x['epoch']==site.epoch()])==10
    print('WAVE4_DISCOVERY_COMPLETE10',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','freeze','qualify','discover']);args=ap.parse_args();r.no_core()
    if args.stage=='prepare':prepare();return
    with r.locked():
        if args.stage=='freeze':freeze()
        else:
            site=r.Site(ROOT);assert site.manifest['phase']=='host_fallback_guard'
            qualify(site) if args.stage=='qualify' else discover(site)

if __name__=='__main__':main()

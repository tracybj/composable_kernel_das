"""Same-binary formal-entry experiments, restoring the paused build on exit."""
import argparse,json,os,shutil,statistics,uuid
from pathlib import Path
import runtime as r
import plan

ROOT=Path(__file__).resolve().parent
BENCH=r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
ARCHIVE=Path('/workspace/v7_coverage/p6h_decode_arch/trial03/benchmark')
ENABLE='CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_WAVE'
EXPERIMENT='CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT'
FIXED_BASELINES={
 'v7':(5000,{}),
 'k16':(5003,{'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M':'128'}),
 'drain':(5007,{EXPERIMENT:'1'}),
 'k32':(5008,{EXPERIMENT:'2'}),
}

def active(x):return x['M']<=64 and x['N']%128==0 and x['K']>=128 and x['K']%128==0

def spec(x,arm,kind,key):
    if arm=='base':expected,env=x['expected'],dict(x['baseline'])
    elif arm=='new':
        expected,env=(6000 if active(x) else x['expected']),dict(x.get('baseline',{}))
        env[ENABLE]='1'
    else:
        expected,env=FIXED_BASELINES[arm];env=dict(env)
        expected=json.loads((ROOT/'selector_expectations.json').read_text())[','.join(str(x[k]) for k in ['M','N','K','G'])][arm]
    timing=kind=='perf'
    command=[str(BENCH),'--dtype=bf16','--layout=NN',
       *[f'--{a}={x[k]}' for a,k in [('m','M'),('n','N'),('k','K'),('groups','G')]],
       '--backend='+('ck' if timing else 'both'),'--blas-lib='+r.BLAS,
       '--include-args-build=0','--validate=1','--warmup=10','--samples=30',
       '--launches-per-sample=1',f'--time={int(timing)}',
       f'--correctness-repeats={1 if timing else 3}',f'--output-redzone={0 if timing else 1}']
    return dict(key=key,id=x['id'],shape={k:x[k] for k in ['M','N','K','G']},
                variant=arm,expected=expected,kind=kind,command=command,env=env)

def execute_group(x,suite,binary):
    arms=['base','new'] if 'baseline' in x else [*FIXED_BASELINES,'new']
    timed=active(x)
    attempt=uuid.uuid4().hex[:10];ep=r.epoch();records=[]
    for arm in arms:
        records.append(r.execute(spec(x,arm,'accuracy',f'{suite}_{x["id"]}_{attempt}_{arm}_acc')))
        assert r.sha(BENCH)==binary
    times={arm:[] for arm in arms};cvs={arm:[] for arm in arms}
    if timed:
        for rnd in range(3):
            ordered=arms[rnd:]+arms[:rnd]
            for arm in ordered:
                rec=r.execute(spec(x,arm,'perf',f'{suite}_{x["id"]}_{attempt}_{arm}_r{rnd+1}'))
                records.append(rec);times[arm].append(rec['result']['median_ms']);cvs[arm].append(rec['result']['cv'])
                assert r.sha(BENCH)==binary
    assert all(z['ok'] and z['epoch']==ep and z['binary_sha256']==binary for z in records)
    row=dict(id=x['id'],suite=suite,shape={k:x[k] for k in ['M','N','K','G']},
       active=active(x),timed=timed,binary_sha256=binary,epoch=ep,physical_gpu=4,
       attempt=attempt,median_ms={arm:statistics.median(v) for arm,v in times.items() if v},
       times=times,cvs=cvs,keys=[z['key'] for z in records],finished_at=r.stamp(),
       instances={arm:next(z['selected'] for z in records if z['variant']==arm) for arm in arms})
    r.append(ROOT/'shape_results.jsonl',row)
    print('SCOPE_CASE_PASS',json.dumps(dict(id=x['id'],suite=suite,median_ms=row['median_ms'])),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['discovery','confirmation']);args=ap.parse_args()
    r.no_core();frozen=json.loads((ROOT/'scope_plan.json').read_text())
    audit=json.loads((ROOT/'binary_audit.json').read_text())
    binary=frozen['binary_sha256'];assert r.sha(ARCHIVE)==binary
    if args.suite=='discovery':
        suites=[('existing',frozen['existing']),('unseen',frozen['unseen_probes']),('boundary',frozen['boundaries'])]
    else:
        choice=json.loads((ROOT/'confirmation_plan.json').read_text())
        suites=[('confirmation',choice['cases'])]
    done=[json.loads(x) for x in (ROOT/'shape_results.jsonl').read_text().splitlines()] if (ROOT/'shape_results.jsonl').exists() else []
    backup=ROOT/'official_benchmark_before_scope'
    with r.locked():
        r.gate('scope_entry_'+args.suite)
        assert r.sha(BENCH)==audit['saved_current_binary_sha256']
        if not backup.exists():shutil.copy2(BENCH,backup)
        assert r.sha(backup)==audit['saved_current_binary_sha256']
        temporary=ROOT/'archived_official_benchmark.install'
        shutil.copyfile(ARCHIVE,temporary);temporary.chmod(BENCH.stat().st_mode & 0o777)
        lease=dict(at=r.stamp(),state='installing_archived_official_benchmark',original_sha256=r.sha(BENCH),experiment_sha256=binary,original_backup=str(backup),entry=str(BENCH))
        (ROOT/'binary_lease.json').write_text(json.dumps(lease,indent=2))
        os.replace(temporary,BENCH)
        try:
            for suite,cases in suites:
                for x in cases:
                    if any(z['suite']==suite and z['id']==x['id'] and z['binary_sha256']==binary for z in done):continue
                    execute_group(x,suite,binary)
        finally:
            assert r.sha(BENCH)==binary,'External binary mutation: do not overwrite another build'
            restore=ROOT/'original_official_benchmark.restore'
            shutil.copy2(backup,restore);os.replace(restore,BENCH)
            assert r.sha(BENCH)==audit['saved_current_binary_sha256']
            lease.update(state='original_benchmark_restored',restored_at=r.stamp())
            (ROOT/'binary_lease.json').write_text(json.dumps(lease,indent=2))
            print('ORIGINAL_FORMAL_BENCHMARK_RESTORED',r.sha(BENCH),flush=True)
    print('SCOPE_SUITE_PASS',args.suite,flush=True)

if __name__=='__main__':main()

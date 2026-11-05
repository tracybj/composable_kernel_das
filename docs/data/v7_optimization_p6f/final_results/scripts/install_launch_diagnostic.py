"""Install only after all old-binary GPU campaigns naturally finish."""
import json
import runtime as r

with r.locked():
    assert not r.MARK.exists() and not r.STOP.exists()
    assert r.remaining(new=True)>600
    meta=json.loads((r.ROOT/'benchmark_launch_diagnostic_pending.json').read_text())
    source=r.REPO/'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args_benchmark.cpp'
    pending=r.ROOT/'benchmark_launch_diagnostic_pending.cpp'
    assert r.sha(source)==meta['source_sha256'] and r.sha(pending)==meta['prepared_source_sha256']
    source.write_bytes(pending.read_bytes())
    for name in ['runtime.py','campaign.py']:
        f=r.ROOT/name;backup=r.ROOT/('pre_launch_diagnostic_'+r.sha(f)+'_'+name)
        assert not backup.exists();backup.write_bytes(f.read_bytes())
    f=r.ROOT/'runtime.py';s=f.read_text()
    needle="    key=spec['key']; assert re.fullmatch(r'[A-Za-z0-9_.-]+',key)"
    insert="""    if spec.get('kind') in ['perf','diagnostic_perf']:
        launches=int(next((a.split('=',1)[1] for a in spec['command'] if a.startswith('--launches-per-sample=')),'1'))
        if spec['kind']=='perf': assert launches==1,'Formal timing must use one launch per sample'
        else: assert launches in [1,20] and launches==spec['launches_per_sample']
"""
    assert s.count(needle)==1;s=s.replace(needle,insert+needle)
    needle="    if spec.get('kind')=='perf':"
    assert s.count(needle)==1;s=s.replace(needle,"    if spec.get('kind') in ['perf','diagnostic_perf']:")
    f.write_text(s)
    f=r.ROOT/'campaign.py';s=f.read_text();needle="in ['perf','native_perf']";assert s.count(needle)==3
    s=s.replace(needle,"in ['perf','native_perf','diagnostic_perf']");f.write_text(s)
    meta.update(installed=True,installed_at=r.stamp(),note='Only benchmark adds explicit diagnostic batch; formal default1 and public API per launch remain unchanged; no new kernel/ABI/provider change')
    (r.ROOT/'launch_diagnostic_install.json').write_text(json.dumps(meta,indent=2))
print('LAUNCH_DIAGNOSTIC_INSTALLED_BUILD_REQUIRED')

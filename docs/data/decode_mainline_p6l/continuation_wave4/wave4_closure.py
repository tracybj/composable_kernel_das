"""Validate explicit completion after the retained sparse-trace stop."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require_closed():
    if read(ROOT/'post_wave4_build_status.json')['state']=='complete' and read(ROOT/'wave4_controller_status.json')['state']=='complete':return
    receipt=read(ROOT/'wave4_phase_closed.json');assert receipt['state']=='complete'
    assert receipt['reason']=='explicit_bounded_sparse_trace_location_repair'
    assert not any((ROOT/n).exists() for n in ['gpu_stop.json','environment_recovery_required.json'])
    assert receipt['epoch']==read(ROOT/'environment_epoch.json')['id']
    assert receipt['benchmark_sha256']==sha('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark')
    for name,digest in receipt['evidence_sha256'].items():assert sha(ROOT/name)==digest
    assert len(read(ROOT/'wave4_qualification.json'))==35
    assert len(read(ROOT/'wave4_discovery_results.json'))==27
    captures=read(ROOT/'wave4_profiles.json')
    assert len(captures)==16 and all(x['raw_token_bytes']>=4096 for x in captures)
    assert read(ROOT/'wave4_discovery_analysis.json')['complete_phase']

#!/usr/bin/env python3
"""Audit source identity, minimum choice, aggregation, coverage and selector availability."""
import collections,csv,gzip,hashlib,json,math,statistics
from pathlib import Path

P=Path(__file__).resolve().parent
def read_gz(name):
    with gzip.open(P/name,'rt') as source:return [json.loads(line) for line in source]
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def main():
    observations=read_gz('all_observations.jsonl.gz');candidates=read_gz('all_candidate_series.jsonl.gz');excluded=read_gz('excluded_records.jsonl.gz')
    best=json.loads((P/'best_known_1193.json').read_text());current=list(csv.DictReader((P/'best_current_1193.csv').open()))
    raw={row['observation_id']:row for row in observations};by_id=collections.defaultdict(list)
    for candidate in candidates:
        rows=[raw[key] for key in candidate['observation_ids']]
        assert all(row['ID']==candidate['ID'] and row['scheme']==candidate['scheme'] for row in rows)
        groups=collections.defaultdict(list)
        for row in rows:groups[row['round']].append(row['ms'])
        assert close(candidate['ms'],statistics.median(statistics.median(values) for values in groups.values()))
        if candidate['gpu']==4 and candidate['protocol']=='card4_device_args_single_launch_warmup10_samples30_args0':by_id[candidate['ID']].append(candidate)
    assert len(best)==len(current)==len(by_id)==1193
    assert [row['ID'] for row in best]==list(range(1,1194))
    assert collections.Counter(row['phase'] for row in best)==dict(decode=881,prefill=312)
    for row,now in zip(best,current):
        choices=by_id[row['ID']];winner=min(choices,key=lambda choice:choice['ms'])
        assert close(winner['ms'],row['best_ms'])
        assert any(choice['candidate_id']==row['best_candidate_id'] and close(choice['ms'],row['best_ms']) for choice in choices)
        runnable=min(choice['ms'] for choice in choices if choice['current_available'])
        assert close(float(now['best_ms']),runnable) and close(row['best_current_ms'],runnable)
        assert close(row['R_frozen'],row['BLAS_ms']/row['best_ms'])
    shape_by_id={int(row['ID']):tuple(int(row[key]) for key in ['M','N','K','G']) for row in csv.DictReader(open('/workspace/shapes_1178.csv',encoding='utf-8-sig'))}
    assert all(tuple(row[key] for key in ['M','N','K','G'])==shape_by_id[row['ID']] for row in observations)
    invalid=set(json.loads(Path('/workspace/v7_coverage/p6e/invalidations.json').read_text())['raw_keys'])
    assert len(invalid)==33
    assert not any(row['source'].endswith('p6e/jobs.jsonl') and row['source_key'] in invalid for row in observations)
    assert {row['key'] for row in excluded if row['reason']=='explicit_environment_invalidation'}==invalid
    assert sum(row['campaign'].startswith('p6e/') for row in observations)==4560
    sources=json.loads((P/'source_catalog.json').read_text())
    for source in sources:assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
    selector=json.loads((P/'selector_audit.json').read_text());checks=list(csv.DictReader((P/'selector_checks.csv').open()))
    assert selector['selector_sha256']==hashlib.sha256(Path('/workspace/composable_kernel/include/ck_tile/ops/gemm/grouped_gemm_selector.hpp').read_bytes()).hexdigest()
    expected={(row['ID'],row['scheme']) for rows in by_id.values() for row in rows if row['current_available']}
    assert selector['passed'] and {(int(row['ID']),row['scheme']) for row in checks}==expected
    assert all(row['expected']==row['actual'] for row in checks)
    summary=json.loads((P/'summary.json').read_text())
    for bucket in summary['buckets']:
        assert sum(bucket[key] for key in ['R_ge099','R_08_099','R_05_08','R_03_05','R_02_03','R_lt02'])==bucket['total']
    result=dict(passed=True,cpu_only=True,shape_count=len(best),candidate_series=len(candidates),observations=len(observations),
        source_files_verified=len(sources),mandatory_invalidations_excluded=len(invalid),selector_cases=len(checks),
        checks=['1193 consecutive unique IDs and actual MNKG','881decode/312prefill','every candidate median reconstructs from recorded rounds',
        'each primary winner is the minimum compatible candidate series','each current winner is the minimum CPU-selector-available series',
        'no fused20/card0 candidate in main choice','33P6e invalidations absent and retained in exclusions','all frozen source SHA256 unchanged','R and mutually exclusive bucket totals verified'])
    (P/'validation.json').write_text(json.dumps(result,indent=2));print('BEST_KNOWN_AUDIT_PASS',json.dumps(result))

if __name__=='__main__':main()

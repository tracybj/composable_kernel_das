#!/usr/bin/env python3
"""Commit-ready own analysis and normalized ledgers, never GPU binaries/BLAS ISA."""
import hashlib,json,shutil,subprocess
from pathlib import Path

P=Path(__file__).resolve().parent
R=Path('/workspace/composable_kernel')
D=R/'docs/data/v7_best_known'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert json.loads((P/'validation.json').read_text())['passed']
    files=['build.py','selector_audit.py','selector_probe.cpp','validate.py','report.py','archive.py',
        'best_known_1193.csv','best_known_1193.json','best_current_1193.csv','best_typical_1193.csv',
        'all_candidate_series.csv','all_candidate_series.jsonl.gz','all_observations.jsonl.gz','excluded_records.jsonl.gz',
        'other_protocol_reference.csv','unknown_protocol_ck_reference.csv','source_catalog.json','source_catalog.csv',
        'shape_label_remaps.json','summary.json','buckets.csv','selector_checks.csv','selector_audit.json','validation.json','report.md']
    D.mkdir(parents=True,exist_ok=True)
    for name in files:shutil.copy2(P/name,D/name)
    manifest=dict(analysis_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        cpu_only=True,no_kernel_or_selector_changes=True,no_gpu_binary_or_blas_isa_copied=True,
        artifacts={name:dict(bytes=(D/name).stat().st_size,sha256=sha(D/name)) for name in files},
        historical_sources='source_catalog.json; original records retained at their recorded paths',
        protected_user_diff_md5=hashlib.md5(subprocess.check_output(['git','diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'],cwd=R)).hexdigest())
    assert manifest['protected_user_diff_md5']=='4c7f49a5c7592767d53195a713db6b8a'
    (D/'manifest.json').write_text(json.dumps(manifest,indent=2));(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('BEST_KNOWN_ARCHIVE_PASS',len(files),'artifacts',sum((D/name).stat().st_size for name in files),'bytes')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Own scripts/derived data/logs only; exclude raw BLAS ISA/code objects."""
import gzip,hashlib,json,os,shutil,stat,subprocess
from pathlib import Path
import run as r
P=r.ROOT;R=r.REPO;D=R/'docs/data/v7_optimization_p6e'
def main():
 audit=json.loads((P/'final_audit.json').read_text());assert audit.get('summary',{}).get('passed')
 D.mkdir(parents=True,exist_ok=True)
 scripts=['run.py','health.py','journalview.py','observe_environment.py','small_run.py','run_native.py','resume_small.py','expand_small.py','diagnose_square.py','scope_analysis.py','decide.py','after_cache.py','analyze_cache.py','analyze_all.py','cache_isa.py','audit_abi.py','selector_equivalence.py','final_audit.py','write_report.py','archive.py','run_probe_v2.py']
 data=['report.md','cache_checkpoint.md','session.json','inputs.json','small_binary.json','cache_isa.json','small_resources.json','cache_times.json','cache_times.csv','cache_profiles.json','cache_raw_counters.json','cache_expanded.json','cache_expanded.csv','cache_walk.json','cache_walk.csv','small_representatives.json','small_representatives.csv','small_expanded.json','small_profiles.json','small_raw_counters.json','native_reference.json','native_reference.csv','new15.json','new15.csv','summary.json','final_audit.json','abi_audit.json','selector_equivalence.json','uncertainty_policy.json','experiment_decisions.json','selector_cpu.log','abi_initial_check_failure.txt','cache_expand_plan.json','small_perf_plan.json','invalidations.json','environment_stop.json','environment_epoch.json','cache_acceptance_checkpoint.json','environment_safety_checkpoint.json','environment_transitions.json','failure_classifications.json','kernel_environment_events.log']
 for name in ['small_precheck.cpp','small_precheck_v2.cpp','probe_build_commands.json','probe_link_command.json','probe_square_build_command.json','probe_k32n64_command.json','probe_v2_link_command.json','small_expanded.csv','drain_wait_model.md','address_decomposition.json','force_final_small_recompile.json','final_isa.out']:
  if (P/name).exists():data.append(name)
 for name in ['per_shape_results','all_regressions','scope_subgroups']:
  data.extend([name+'.json',name+'.csv'])
 data.extend(['final_environment_observation.json','kernel_environment_final.log'])
 data.extend(['square_diagnostic_plan.json','square_diagnostic_profiles.json'])
 for p in P.glob('small_expand_plan_*.json'):data.append(p.name)
 for p in P.glob('small_expand_*.out'):data.append(p.name)
 for pattern in ['expand_*.out','*_rec3.out','recovery*_resolved.json']:
  for p in P.glob(pattern):
   if p.name not in data:data.append(p.name)
 for name in scripts+data:
  source=P/name
  if not source.exists():raise RuntimeError('Missing required artifact '+str(source))
  shutil.copy2(source,D/name)
 # Git normalizes CSV text on checkout. Pin the actual committed bytes while
 # preserving all numeric values and every raw log byte in the gzip records.
 for path in D.glob('*.csv'):
  path.write_text(path.read_text(),newline='\n')
 # Durable journal + exact outputs; no .perf or code-object payloads in Git.
 with gzip.open(D/'journals.json.gz','wt',encoding='utf-8') as f:
  json.dump({name:(P/name).read_text() if (P/name).exists() else None for name in ['jobs.jsonl','gate.jsonl','waits.jsonl','recovery.jsonl']},f)
 records=[json.loads(x) for x in (P/'jobs.jsonl').read_text().splitlines()]
 if (P/'recovery.jsonl').exists():records += [json.loads(x) for x in (P/'recovery.jsonl').read_text().splitlines()]
 with gzip.open(D/'logs.jsonl.gz','wt',encoding='utf-8') as f:
  for x in records:
   folder=Path(x['path']);f.write(json.dumps(dict(key=x['key'],path=x['path'],run_log=(folder/'run.log').read_text(errors='replace'),command_json=(folder/'command.json').read_text() if (folder/'command.json').exists() else None),ensure_ascii=False)+'\n')
 with gzip.open(D/'build_logs.json.gz','wt',encoding='utf-8') as f:
  json.dump({p.name:p.read_text(errors='replace') for p in P.glob('build*.log')},f)
 with gzip.open(D/'prototypes.json.gz','wt',encoding='utf-8') as f:
  json.dump(dict(note='Own CK-derived historical/deferred prototypes only. Not linked into the final benchmark; no performance claim for A-index fold.',files={p.name:p.read_text() for p in (P/'prototypes').glob('*.hpp')}),f)
 with gzip.open(D/'script_history.json.gz','wt',encoding='utf-8') as f:
  json.dump({p.name:p.read_text() for p in (P/'scripts').glob('*.py')},f)
 # Preserve executable modes for local recovery/reproduction backups.
 for name in ['bench_before','bench_cache','bench_small']:
  p=P/name
  if p.exists():p.chmod(stat.S_IMODE(r.BENCH.stat().st_mode))
 sourcefiles=set(subprocess.check_output(['git','diff','83c8c2a','--name-only'],cwd=R,text=True).splitlines())|set(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=R,text=True).splitlines())
 sourcefiles={name for name in sourcefiles if name.endswith(('.cpp','.hpp','.h')) or name.endswith('CMakeLists.txt') or 'provider_sources.txt' in name}
 manifest=dict(source_base='83c8c2a787d9994c7309feb034be705c3f3e6203',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),source_files={name:r.sha(R/name) for name in sorted(sourcefiles)},binary_files={name:r.sha(P/name) for name in ['bench_before','bench_cache','bench_small']},artifacts={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=r.sha(p)) for p in sorted(D.iterdir()) if p.is_file() and p.name!='manifest.json'},raw_data_retained_outside_git=str(P),raw_blas_isa_not_copied=True,csv_newline_policy='Canonical LF for committed artifacts; numeric values unchanged; raw logs preserved in gzip')
 (D/'manifest.json').write_text(json.dumps(manifest,indent=2));(P/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('ARTIFACT_ARCHIVE_PASS',len(manifest['artifacts']),'files',sum(p.stat().st_size for p in D.iterdir() if p.is_file()),'bytes')
if __name__=='__main__':main()

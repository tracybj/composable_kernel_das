#!/usr/bin/env python3
"""Archive compact derived evidence; never archive vendor code/disassembly."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
DEST=REPO/'docs/data/v7_blas_isa_p6d'

def main():
 v=json.loads((ROOT/'validation.json').read_text())
 assert v['timing_jobs']==75 and v['profile_jobs']==13 and v['trace_jobs']>=1,v
 assert v['jobs']==v['successful']==89 and not v['failures'],v
 assert v['all_profile_mmop_matched'] and v['all_isa_decoded']
 gates=[json.loads(s) for s in (ROOT/'gate.jsonl').read_text().splitlines()]
 assert len(gates)==89 and all(r['attempt']==1 and r['rc']==0 and r['hcu4']=='0.0%' for r in gates)
 for name,expected in json.loads((ROOT/'manifest.json').read_text()).items():
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected['sha256'],name
 protected='include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'
 diff=subprocess.check_output(['git','diff','--',protected],cwd=REPO)
 assert hashlib.md5(diff).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
 assert Path('/workspace/shapes_1178.csv').read_bytes()==Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv').read_bytes()
 shapes_sha=hashlib.sha256(Path('/workspace/shapes_1178.csv').read_bytes()).hexdigest()
 assert shapes_sha=='5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863'
 (ROOT/'post_audit.json').write_text(json.dumps(dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),protected_diff_md5=hashlib.md5(diff).hexdigest(),shape_sha256=shapes_sha,all_pinned_inputs_unchanged=True),indent=2))
 paths=[ROOT/n for n in ['run.py','blas_replay.py','elf_isa.py','isa_features.py','analyze.py','write_report.py','summarize_trace.py','archive.py']]
 paths += [Path(p) for p in ['/opt/dtk/aillvm/bin/llvm-objdump','/opt/rocm-6.3.3/bin/xprof','/opt/rocm-6.3.3/lib/libxprofiler.so.3.16.0','/workspace/v7_coverage/p4a/perfdump.py','/workspace/v7_coverage/p6/analyze.py','/workspace/Grouped_Geem/blaslt_tuning_tools/build/libgroupgemm_tune.so','/workspace/Grouped_Geem/blaslt_tuning_tools/groupgemm_bridge.cpp','/workspace/Grouped_Geem/blaslt_tuning_tools/tuning_common.py','/workspace/v7_coverage/p6/summary.json','/workspace/v7_coverage/p6b/stage_wait/profile_summary.json']]
 (ROOT/'script_tool_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
 DEST.mkdir(parents=True,exist_ok=True)
 names=['report.md','design.md','cases.json','manifest.json','isa_features.json','coverage.json','timing.json','profiles.json','raw_counters.json','historical_profiles.json','v4_fallback_metadata.json','validation.json','post_audit.json','script_tool_sha256.json','trace_summary.json','run.py','blas_replay.py','elf_isa.py','isa_features.py','analyze.py','write_report.py','summarize_trace.py','archive.py']
 for name in names:shutil.copy2(ROOT/name,DEST/name)
 for name in ['jobs.jsonl','gate.jsonl']:
  (DEST/(name+'.gz')).write_bytes(gzip.compress((ROOT/name).read_bytes(),compresslevel=9,mtime=0))
 jobs=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
 logs=[dict(key=r['key'],ok=r.get('ok'),log=(Path(r['path'])/'run.log').read_text(errors='replace')) for r in jobs]
 (DEST/'logs.json.gz').write_bytes(gzip.compress(json.dumps(logs,ensure_ascii=False).encode(),compresslevel=9,mtime=0))
 shutil.copy2(ROOT/'analysis.out',DEST/'analysis.txt')
 (DEST/'README.md').write_text('''# P6d BLAS ISA-guided diagnosis

结论与下一阶段候选见 [report.md](report.md)。九个代表shape，包含新增M128/M256及大M大K；不是1193全量。本阶段没有修改CK实现和默认selector。

`isa_features.json`只保存自己的解析结果、资源元数据、符号/哈希和控制流计数，不包含BLAS机器码或反汇编正文；raw `.s`、code object、`.perf`/SQTT在`/workspace/v7_coverage/p6d/`。`coverage.json`说明选定六个BLAS内核在历史1193条映射中的覆盖，不表示对724条做了GPU复测。`timing.json`保留每轮中位数、CV和全局范围；Native repeat1/20分别列出。`profiles.json`是本次13组计数，`historical_profiles.json`为明确标注的P6a/P6b证据复用。`raw_counters.json`保留每个有效profile的两个dispatch。

`jobs.jsonl.gz`/`gate.jsonl.gz`/`logs.json.gz`为完整命令、门禁、结果与日志的无损压缩。现场依赖由manifest和script_tool_sha256固定；脚本是现场环境审计副本，重放需原路径的工具/BLAS/benchmark。没有复制BLAS实现进CK；没有push。
''')
 print('ARCHIVE_PASS',DEST,'files',len(list(DEST.iterdir())),'bytes',sum(p.stat().st_size for p in DEST.iterdir()))

if __name__=='__main__':main()

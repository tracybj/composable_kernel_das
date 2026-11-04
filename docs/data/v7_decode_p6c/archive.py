#!/usr/bin/env python3
"""Copy compact, reviewable final evidence after the campaign passes."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
DEST=REPO/'docs/data/v7_decode_p6c'

def main():
 result=json.loads((ROOT/'validation.json').read_text())
 assert result['valid_jobs']==3759
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for name,spec in manifest.items():
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==spec['sha256'],name
 assert Path('/workspace/shapes_1178.csv').read_bytes()==Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv').read_bytes()
 protected='include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'
 protected_diff=subprocess.check_output(['git','diff','--',protected],cwd=REPO)
 assert hashlib.md5(protected_diff).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
 receipt=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
  protected_diff_md5=hashlib.md5(protected_diff).hexdigest(),input_files_unchanged=True,
  benchmark_sha256=manifest[str(REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark')]['sha256'])
 (ROOT/'post_audit.json').write_text(json.dumps(receipt,indent=2))
 scripts=[ROOT/n for n in ['run.py','analyze.py','write_report.py','preview.py','archive.py']]
 scripts.append(Path('/workspace/Grouped_Geem/blaslt_tuning_tools/tuning_common.py'))
 (ROOT/'script_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},indent=2))
 DEST.mkdir(parents=True,exist_ok=True)
 names=['report.md','design.md','plan.json','manifest.json','script_sha256.json','post_audit.json',
  'validation.json','tables.json','measured.csv','measured.json','merged1193.csv','native_new15.json','environment_failure.json',
  'run.py','analyze.py','write_report.py','preview.py','archive.py']
 for name in names:shutil.copy2(ROOT/name,DEST/name)
 for name in ['jobs.jsonl','gate.jsonl']:
  (DEST/(name+'.gz')).write_bytes(gzip.compress((ROOT/name).read_bytes(),compresslevel=9,mtime=0))
 failed=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines() if not json.loads(s)['ok']]
 for number,record in enumerate(failed,1):
  shutil.copy2(Path(record['path'])/'run.log',DEST/f'environment_failure_attempt{number}.log')
 for number in [1,2,3]:
  path=ROOT/f'round{number}_preview.json'
  if path.exists():shutil.copy2(path,DEST/path.name)
 shutil.copy2(ROOT/'analysis.out',DEST/'analysis.txt')
 shutil.copy2(ROOT/'new_blas/run_config.json',DEST/'new_blas_config.json')
 env=ROOT/'new_blas/cases/1179/environment.json'
 if env.exists():shutil.copy2(env,DEST/'new_blas_environment.json')
 tables=json.loads((ROOT/'tables.json').read_text())
 flat=[dict(dataset=r['dataset'],phase=r['phase'],total_case=r['total_case'],**dict(zip(
  ['R_ge_099','R_08_to_099','R_05_to_08','R_03_to_05','R_02_to_03','R_lt_02'],r['buckets']))) for r in tables]
 with (DEST/'buckets.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
 (DEST/'README.md').write_text('''# P6c 小 M 增量分桶评估

完整结论、互斥分桶和新增15条明细见 [report.md](report.md)。本阶段只测615个不同CK shape，不做1193全量；原578条沿用11-03。主方案模拟M≤64启用Tn128（5003），默认开关未提升。报告完整保留回退，未逐条挑选快者。

`measured.csv/JSON`保存618个原型对照（原600条+新增9条各两种tile），其中607条纳入主策略。`merged1193.csv`逐条记录baseline/after、R、实例和来源。新增6条prefill只有原路径基线，标为current_new_baseline；其余实测标为current_AB，历史沿用标为11-03_carried。`buckets.csv`保留所有对照表。

原1178的BLAS来自卡0冻结数据，新增15条是卡4Native GroupedGemm按原协议补测，因此扩展1193表含不同BLAS来源，属于参考拼接表。CK前后A/B均卡4；新增M128/256的原型对照不加入主策略。

`jobs.jsonl.gz`和`gate.jsonl.gz`为完整逐条命令、结果和门禁的无损压缩。原始logs、Native候选数据及冻结依赖留在 `/workspace/v7_coverage/p6c/` 和manifest列出的现场路径。脚本是现场环境的审计副本，重放需对应的旧/新benchmark与BLAS依赖。Native配置继承冻结基线协议模板，实际执行时间以jobs及每case的finished_at为准。

分析脚本严格要求3759项成功（accuracy18/newblas15/perf3726），核对分桶和、截图11-03原值及来源。`post_audit.json`确认数据/二进制哈希不变、readfirstlane差异仍保留。报告中只有本次新增decode的54次比较称为CK全输出验证；其他CK计时为CPU抽样检查。
''')
 print('ARCHIVE_PASS',DEST,'files',len(list(DEST.iterdir())))

if __name__=='__main__':main()

"""CPU-only validation and packaging of the completed ID76 paired capture."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path('/workspace/composable_kernel')
DOC = REPO / 'docs/data/v7_decode_wave_p6h/sqtt_id76_20261109'
sys.path.insert(0, '/workspace/v7_coverage/p4a')
from perfdump import dispatches


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = json.loads((HERE / 'result.json').read_text())
    final = json.loads((HERE / 'fixed_cu_result.json').read_text())
    assert original['ok'] and final['ok']
    DOC.mkdir(parents=True, exist_ok=False)
    records = original['jobs'] + final['jobs']
    keys = {x['key'] for x in records}
    accuracy = [x for x in records if x['kind'] == 'accuracy']
    assert len(accuracy) == 2 and len(final['jobs']) == 2
    assert {x['selected'] for x in accuracy} == {5008, 6000}
    for x in accuracy:
        assert len(x['comparisons']) == 3
        assert all(int(y[1]) == 8192 and int(y[2]) == 0 and float(y[3]) == 0 for y in x['comparisons'])
    gates = [json.loads(line) for line in (ROOT / 'gate.jsonl').read_text().splitlines()
             if json.loads(line)['key'].startswith('sqtt76_')]
    kernel = [json.loads(line) for line in (ROOT / 'kernel_checks.jsonl').read_text().splitlines()
              if json.loads(line)['key'].startswith('sqtt76_')]
    assert all(x['rc'] == 0 and x['hcu'] == '0.0%' and 'MKFD' not in x['output'].upper() for x in gates)
    assert all(x['rc'] == 0 and not x['events'] for x in kernel)
    assert all(any(g['key'] == key for g in gates) for key in keys)
    assert all(x['ok'] and x['physical_gpu'] == 4 and not x['kernel_service_events'] for x in records)
    assert len({x['epoch'] for x in records}) == 1
    assert sha(original['plan']['binary']) == original['plan']['binary_sha256'] == final['binary_sha256']
    diff = subprocess.check_output(['git', '-C', str(REPO), 'diff', '--',
                                   'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'])
    assert hashlib.md5(diff).hexdigest() == '4c7f49a5c7592767d53195a713db6b8a'
    shapes = {}
    for path in ['/workspace/shapes_1178.csv', '/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv']:
        shapes[path] = sha(path)
        assert shapes[path] == '5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863'
    expected = {'base': {'SQ_WAVES': 256, 'SQ_INSTS_VMEM': 22400, 'SQ_INSTS_LDS': 46080,
                         'SQ_INSTS_VALU': 159360, 'SQ_INSTS_MMOP': 98304},
                'new': {'SQ_WAVES': 256, 'SQ_INSTS_VMEM': 104704, 'SQ_INSTS_LDS': 14592,
                        'SQ_INSTS_VALU': 169728, 'SQ_INSTS_MMOP': 24576}}
    aliases = {'base': 'ID76_baseline_K32_SQTT.perf', 'new': 'ID76_A_LDS_B_REG_SQTT.perf'}
    profiles = []
    for job in final['jobs']:
        profile = Path(job['perf_files'][0])
        ds = dispatches(profile)
        assert len(ds) == 1 and len(ds[0]) == 199
        assert {k: ds[0][k] for k in expected[job['variant']]} == expected[job['variant']]
        text = (Path(job['path']) / 'profile/results.txt').read_text()
        assert 'dispatch[18], gpu-id(4)' in text
        reported = re.findall(r'SE\((\d+)\) size\((\d+)\)', text)
        raw = sum(x.get('raw_token_bytes', 0) for x in job['capture_files'])
        assert sum(int(n) for _, n in reported) == raw and raw >= 4096
        assert any(x['path'].endswith('_se0_cu1.out') for x in job['capture_files'])
        alias = HERE / aliases[job['variant']]
        shutil.copyfile(profile, alias)
        assert sha(profile) == sha(alias)
        profiles.append(dict(variant=job['variant'], instance=job['selected'], key=job['key'],
                             original_perf=str(profile), download_perf=str(alias), bytes=alias.stat().st_size,
                             sha256=sha(alias), raw_sqtt_bytes=raw, target_dispatch=18,
                             scope='SE0 / hardware CU1 (active CU index 0)',
                             counters=expected[job['variant']], counter_records=len(ds[0])))
    with (ROOT.parent / 'p6b/run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    running = []
    for x in records:
        path = Path('/proc') / str(x['pid']) / 'cmdline'
        if path.exists():
            running.append(dict(pid=x['pid'], cmdline=path.read_bytes().replace(b'\0', b' ').decode()))
    assert not running, running
    summary = dict(ok=True, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   task='ID76 paired PMC + SQTT capture, no kernel changes', physical_gpu=4,
                   epoch=final['epoch'], binary_sha256=final['binary_sha256'], profiles=profiles,
                   raw_gpu_jobs=6, accuracy_jobs=2, full_output_comparisons=6,
                   full_output_elements=sum(int(y[1]) for x in accuracy for y in x['comparisons']),
                   failed_elements=0, max_abs=0, healthy_checks=len(gates), kernel_checks=len(kernel),
                   kernel_service_events=[], environment_recovery_required=False,
                   initial_sparse_baseline_retained=True, instruction_latencies_decoded_locally=False,
                   windows_gui_verified=False, capture_timing_is_formal=False,
                   kernel_or_selector_changes=False, p6g_state='paused', live_jobs=running,
                   protected_diff_md5=hashlib.md5(diff).hexdigest(), shape_sha256=shapes)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2))
    readme = '''# ID76：A-LDS/B-register 原型与旧 K32 的 SQTT 对照

2026-11-09，仅物理 GPU4 串行；M=1、N=2048、K=768、G=4。

下载文件：
- `ID76_A_LDS_B_REG_SQTT.perf`：trial06b 新原型，实例6000。
- `ID76_baseline_K32_SQTT.perf`：既有小M K32，实例5008。

用 XCompute 打开新的文件，选择 dispatch18、XCD0。通过 Wavefront/Inst 查看所采 wave 的指令时间线，通过 Source 查看对应的指令、Hit Count 和 Latency（以 GUI 实际解码结果为准）。两份文件均含原8类 section的 PMC 和 SQTT；详情参见 summary.json。不要使用此前不带 SQTT 的旧文件。

采样限定 SE0 的首个活动 CU，文件硬件编号为CU1；不能把该CU的指令时间线当成所有CU的全局耗时。Linux侧已核对实际原始token存在、尺寸与xprof结果一致、目标实例/dispatch/PMC正确；本机没有运行Windows XCompute GUI，因此尚未验证GUI的指令匹配或逐指令Latency，不声称已完成动态瓶颈归因。

两个方案先各做3次CK与BLAS全输出比较，均8192元素/次、failed=0、max_abs=0，CK/BLAS redzone均通过。二进制SHA与11月6日的trial06b完全一致。新旧选核、源代码和历史性能表未变。

首次自动选CU时，旧方案仅800字节SQTT；原记录保留，不作为交付的指令分析对照。随后显式固定同一个CU重采双方，得到旧65920字节、新84832字节SQTT。这是采样覆盖修正，没有按性能快慢挑选结果。

采集命令含 `--enable-sqtt --enable-raw --target-se 0 --target-cu 0 --dispatches 18`，并保留 `--sections` 的原8项。每条GPU命令前均通过hy-smi空闲和内核服务门禁，统一锁串行；未发现MKFD/服务事件。没有重新计时、全量测试或恢复已暂停P6g。

特别注意：xprof replay显著改变benchmark事件计时；run.log中的PERF不是正式性能数据，不能据此更新R或分桶。

目录中脚本和command/run.log记录可用于核对命令；`.out`是额外导出的原始16位十六进制token，不是已解码的指令耗时。主下载包包含两份最终.perf、原始SQTT、results.txt、命令、README和summary。
'''
    (HERE / 'README.md').write_text(readme)
    for name in ['capture.py', 'capture_fixed_cu.py', 'archive.py', 'plan.json', 'result.json',
                 'fixed_cu_result.json', 'summary.json', 'README.md']:
        shutil.copyfile(HERE / name, DOC / name)
    for name in ['runtime.py', 'health.py', 'campaign.py']:
        shutil.copyfile(ROOT / name, DOC / name)
    for job in records:
        out = DOC / 'runs' / job['key']
        out.mkdir(parents=True)
        for name in ['command.json', 'run.log']:
            shutil.copyfile(Path(job['path']) / name, out / name)
        results = Path(job['path']) / 'profile/results.txt'
        if results.exists():
            shutil.copyfile(results, out / 'results.txt')
    for name, values in [('gate.jsonl', gates), ('kernel_checks.jsonl', kernel)]:
        (DOC / name).write_text(''.join(json.dumps(x) + '\n' for x in values))
    bundle = HERE / 'ID76_SQTT_comparison.zip'
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*aliases.values(), 'README.md', 'summary.json']:
            archive.write(HERE / name, name)
        for job in final['jobs']:
            folder = Path(job['path'])
            for name in ['command.json', 'run.log']:
                archive.write(folder / name, job['variant'] + '/' + name)
            for p in sorted((folder / 'profile').iterdir()):
                if p.is_file() and p.suffix != '.perf':
                    archive.write(p, job['variant'] + '/' + p.name)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        for name in aliases.values():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha(HERE / name)
    manifest = {str(p.relative_to(DOC)): sha(p) for p in sorted(DOC.rglob('*')) if p.is_file()}
    (DOC / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (HERE / 'completion_receipt.json').write_text(json.dumps(dict(
        at=summary['finished_at'], ok=True, bundle=str(bundle), bundle_sha256=sha(bundle),
        git_evidence=str(DOC), summary=str(HERE / 'summary.json')), indent=2))
    print('ID76_SQTT_ARCHIVE_PASS', len(manifest), 'files', 'bundle_bytes=' + str(bundle.stat().st_size))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

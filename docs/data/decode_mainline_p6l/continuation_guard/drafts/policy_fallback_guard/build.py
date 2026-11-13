"""Build a host-only adapter repair and prove every GPU shader unchanged."""
from pathlib import Path
import json, os, re, shutil, subprocess, sys, time
D = Path(__file__).resolve().parent
ROOT = D.parents[1]
sys.path.insert(0, str(ROOT))
import plan, runtime as r, shader_audit


def main():
    r.no_core()
    session = r.read(ROOT / 'session.json')
    assert time.time() + 900 < session['gpu_deadline_epoch']
    assert r.read(D / 'policy_equivalence.json')['ok']
    assert r.read(ROOT / 'noslc_finish_status.json')['state'] == 'complete_rejected_candidate'
    assert r.read(ROOT / 'noslc_archive_commit.json')['user_worktree_preserved']
    assert not any((ROOT / x).exists() for x in ['gpu_stop.json', 'environment_recovery_required.json', 'user_pause.json'])
    old = r.read(ROOT / 'frozen_plan.json')
    assert old['benchmark_sha256'] == plan.sha(plan.BENCH)
    name = r.read(D / 'plan.json')['source']
    for path, digest in old['worktree_source_sha256'].items():
        assert plan.sha(D / 'before_worktree.cpp' if path == name else plan.REPO / path) == digest
    folder = ROOT / 'compile/build16'
    assert not folder.exists()
    assert shutil.disk_usage(ROOT).free >= 20 * 1024**3
    folder.mkdir(parents=True)
    names = ['tile_example_grouped_gemm_device_args_benchmark', 'tile_example_grouped_gemm_device_args_query_test',
             'tile_example_grouped_gemm_device_args_smoke', 'tile_example_grouped_gemm_selector_test',
             'tile_example_grouped_gemm_decode_depth8_selector_test', 'tile_example_grouped_gemm_decode_mainline_selector_test']
    backup = folder / 'previous'
    backup.mkdir()
    for item in names:
        shutil.copy2(plan.REPO / 'build/bin' / item, backup / item)
    shutil.copy2(ROOT / 'frozen_plan.json', backup / 'frozen_plan.json')
    reference = shader_audit.inventory(plan.BENCH)
    assert len(reference) == 1766
    r.write(folder / 'reference_shaders.json', dict(binary_sha256=plan.sha(plan.BENCH), kernels=reference))
    sources = {path: plan.sha(plan.REPO / path) for path in old['worktree_source_sha256']}
    assert len(sources) == 26
    r.write(folder / 'source_sha256.json', sources)
    for path, digest in sources.items():
        target = folder / 'compiled_sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.REPO / path, target)
        assert plan.sha(target) == digest
    os.sched_setaffinity(0, {124, 125, 126, 127})
    os.nice(10)
    command = ['cmake', '--build', str(plan.REPO / 'build'), '--target', *names, '-j4']
    r.write(folder / 'controller.json', dict(at=r.stamp(), state='running', pid=os.getpid(), GPU=False, command=command))
    start = time.monotonic()
    env = os.environ.copy()
    env.update(HIP_VISIBLE_DEVICES='4', CUDA_VISIBLE_DEVICES='4')
    try:
        assert int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text())[1]) >= 96 * 1024 * 1024
        with (folder / 'build.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
            r.write(folder / 'live.json', dict(at=r.stamp(), pid=child.pid, command=command, GPU=False))
            rc = child.wait()
        record = dict(rc=rc, seconds=time.monotonic() - start, at=r.stamp(), command=command)
        r.write(folder / 'results.json', [record])
        assert rc == 0
        assert all(plan.sha(plan.REPO / path) == digest for path, digest in sources.items())
        actual = shader_audit.inventory(plan.BENCH)
        assert actual == reference, 'Host-only repair must preserve every shader byte and resource record'
        result = dict(at=r.stamp(), ok=True, CPU_only=True, GPU=False, benchmark_sha256=plan.sha(plan.BENCH),
                      all_shader_inventory_and_bytes_unchanged=1766, source_sha256=sources,
                      selected_instance_ABI_flags_unchanged=True, compiled_seconds=record['seconds'])
        r.write(folder / 'completed.json', result)
        r.write(ROOT / 'guard_build_isa_audit.json', result)
        r.write(folder / 'controller.json', dict(at=r.stamp(), state='complete', pid=os.getpid(), GPU=False))
        print('HOST_GUARD_OFFICIAL_BUILD_ALL1766_SHADERS_UNCHANGED', record['seconds'], result['benchmark_sha256'], flush=True)
    except BaseException as error:
        r.write(folder / 'controller.json', dict(at=r.stamp(), state='stopped', pid=os.getpid(), GPU=False, error=str(error)))
        raise


if __name__ == '__main__':
    main()
